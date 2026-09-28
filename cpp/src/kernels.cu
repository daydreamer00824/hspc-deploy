#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

#include "cuda_utils.h"
#include "kernels.h"
#include "matching_rule.cuh"

static_assert(kNumVariants == kVariants, "kVariants (kernels.h) must equal kNumVariants (matching_rule.cuh)");

// ---------------------------------------------------------------- 标准化 + gather
__global__ void normalize_gather_kernel(const float* __restrict__ cube, int bands, int rows, int cols,
                                        const float* __restrict__ mean, const float* __restrict__ stdv,
                                        const int32_t* __restrict__ vr, const int32_t* __restrict__ vc, int64_t total,
                                        float* __restrict__ out) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int k = static_cast<int>(idx % 9);
  const int64_t t = idx / 9;
  const int b = static_cast<int>(t % bands);
  const int i = static_cast<int>(t / bands);
  const int r = vr[i] + k / 3 - 1, c = vc[i] + k % 3 - 1;
  float v = 0.f;
  if (r >= 0 && r < rows && c >= 0 && c < cols) v = (cube[(static_cast<int64_t>(b) * rows + r) * cols + c] - mean[b]) / stdv[b];
  out[idx] = v;
}

void launch_normalize_gather(const float* cube, int bands, int rows, int cols, const float* mean, const float* stdv,
                             const int32_t* vr, const int32_t* vc, int n, float* patches, cudaStream_t s) {
  const int64_t total = static_cast<int64_t>(n) * bands * 9;
  const int threads = 256;
  normalize_gather_kernel<<<static_cast<unsigned>((total + threads - 1) / threads), threads, 0, s>>>(cube, bands, rows, cols, mean,
                                                                                                    stdv, vr, vc, total, patches);
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------- 特征归一化
template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <>
__device__ __forceinline__ float to_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }

template <typename T>
__global__ void normalize_features_kernel(const T* __restrict__ y, float* __restrict__ out) {
  constexpr int kPer = kFeatDim / 256;
  const T* row = y + static_cast<int64_t>(blockIdx.x) * kFeatDim;
  float v[kPer], ss = 0.f;
  for (int j = 0; j < kPer; ++j) {
    v[j] = to_f32<T>(row[threadIdx.x + j * 256]);
    ss = fmaf(v[j], v[j], ss);
  }
  __shared__ float red[8];
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int w = 0; w < 8; ++w) tot += red[w];
  const float denom = sqrtf(tot) + 1e-12f;
  for (int j = 0; j < kPer; ++j) out[static_cast<int64_t>(blockIdx.x) * kFeatDim + threadIdx.x + j * 256] = v[j] / denom;
}

void launch_normalize_features(const void* y, bool y_is_half, int n, float* out, cudaStream_t s) {
  if (n <= 0) return;
  if (y_is_half) normalize_features_kernel<__half><<<n, 256, 0, s>>>(static_cast<const __half*>(y), out);
  else normalize_features_kernel<float><<<n, 256, 0, s>>>(static_cast<const float*>(y), out);
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------- 融合窗口匹配
constexpr int kMaxWin = 11;  // 2*radius+1 的上限（本项目 radius=5）

__global__ void match_kernel(const float* __restrict__ hsi, const int32_t* __restrict__ lut, const float* __restrict__ pc,
                             const int32_t* __restrict__ ref_rows, const int32_t* __restrict__ ref_cols, int n_points, int rows,
                             int cols, int radius, float inv_radius, int32_t* __restrict__ out_rows, int32_t* __restrict__ out_cols,
                             float* __restrict__ out_cos) {
  const int p = blockIdx.x;
  const int w = 2 * radius + 1, nwin = w * w;
  __shared__ __align__(16) float s_pf[kFeatDim];
  __shared__ float s_cos[kMaxWin * kMaxWin];
  __shared__ int s_valid[kMaxWin * kMaxWin];
  for (int i = threadIdx.x; i < kFeatDim; i += blockDim.x) s_pf[i] = pc[static_cast<int64_t>(p) * kFeatDim + i];
  __syncthreads();

  const int rr0 = ref_rows[p], cc0 = ref_cols[p];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = blockDim.x >> 5;
  for (int cand = warp; cand < nwin; cand += nwarps) {
    const int r = rr0 + cand / w - radius, c = cc0 + cand % w - radius;
    int v = -1;
    if (r >= 0 && r < rows && c >= 0 && c < cols) v = lut[r * cols + c];
    float dot = 0.f;
    if (v >= 0) {
      const float4* g = reinterpret_cast<const float4*>(hsi + static_cast<int64_t>(v) * kFeatDim);
      const float4* q = reinterpret_cast<const float4*>(s_pf);
#pragma unroll
      for (int j = 0; j < kFeatDim / 128; ++j) {
        const float4 a = g[lane + j * 32], b = q[lane + j * 32];
        dot = fmaf(a.x, b.x, dot); dot = fmaf(a.y, b.y, dot); dot = fmaf(a.z, b.z, dot); dot = fmaf(a.w, b.w, dot);
      }
      for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
    }
    if (lane == 0) { s_cos[cand] = dot; s_valid[cand] = v >= 0; }
  }
  __syncthreads();

  if (threadIdx.x < kNumVariants) {
    const int variant = threadIdx.x;
    const float sw = kSpatialWeights[variant];
    float best = -INFINITY;
    int best_i = -1;
    for (int cand = 0; cand < nwin; ++cand) {
      if (!s_valid[cand]) continue;
      const float dy = static_cast<float>(cand / w - radius), dx = static_cast<float>(cand % w - radius);
      const float score = combine_score(s_cos[cand], sqrtf(dy * dy + dx * dx), sw, inv_radius);
      if (score > best) { best = score; best_i = cand; }  // 严格大于：并列取行优先的第一个（与 argmax 一致）
    }
    const int o = variant * n_points + p;
    if (best_i < 0) { out_rows[o] = -1; out_cols[o] = -1; out_cos[o] = nanf(""); }
    else { out_rows[o] = rr0 + best_i / w - radius; out_cols[o] = cc0 + best_i % w - radius; out_cos[o] = s_cos[best_i]; }
  }
}

void launch_match(const float* hsi_feat_norm, const int32_t* lut, const float* pc_feat_norm, const int32_t* ref_rows,
                  const int32_t* ref_cols, int n_points, int rows, int cols, int radius, int32_t* out_rows,
                  int32_t* out_cols, float* out_cos, cudaStream_t s) {
  if (n_points <= 0) return;
  // s_cos/s_valid 按 kMaxWin*kMaxWin 分配，radius 超出这个上限会让 cand 下标越过共享内存数组末尾。
  if (radius < 0 || 2 * radius + 1 > kMaxWin)
    throw std::runtime_error("launch_match: radius " + std::to_string(radius) + " exceeds shared-memory window size (max "
                              + std::to_string((kMaxWin - 1) / 2) + ")");
  const float inv_radius = 1.0f / fmaxf(static_cast<float>(radius), 1.0f);
  match_kernel<<<n_points, 128, 0, s>>>(hsi_feat_norm, lut, pc_feat_norm, ref_rows, ref_cols, n_points, rows, cols, radius, inv_radius,
                                        out_rows, out_cols, out_cos);
  CUDA_CHECK(cudaGetLastError());
}
