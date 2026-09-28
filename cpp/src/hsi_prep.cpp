#include "hsi_prep.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <vector>

#include "threading.h"

// NumPy 的 float32 pairwise 求和（numpy/core/src/umath/loops_utils.h.src 的 pairwise_sum）：
// n<8 顺序累加；n<=128 用 8 个累加器展开；更大则对半分（块边界对齐到 8）。
static float pairwise_sum(const float* a, int64_t n) {
  if (n < 8) {
    float res = 0.f;
    for (int64_t i = 0; i < n; ++i) res += a[i];
    return res;
  }
  if (n <= 128) {
    float r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    int64_t i;
    for (i = 8; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; ++j) r[j] += a[i + j];
    float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  int64_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// np.add.reduce 在一段连续内存上的实际行为：迭代器按 8192 个元素分块（NumPy 的默认缓冲区大小），每块内部做 pairwise，
// 块与块之间从 0 开始顺序累加。用 numpy 1.26.4 在 Jetson 上与 raw.sum(axis=(1,2)) 逐位对过（342 个波段），只有这种口径完全一致。
static float np_sum(const float* a, int64_t n) {
  constexpr int64_t kBuffer = 8192;
  float acc = 0.f;
  for (int64_t i = 0; i < n; i += kBuffer) acc += pairwise_sum(a + i, std::min(kBuffer, n - i));
  return acc;
}

void hsi_valid_mask(const float* raw, int bands, int64_t npix, int red_index, int nir_index, float ndvi_threshold,
                    float nodata_epsilon, uint8_t* mask, int workers, int64_t* out_valid_count) {
  const float* red = raw + red_index * npix;
  const float* nir = raw + nir_index * npix;
  std::atomic<int64_t> count{0};
  parallel_for(npix, workers, [&](int64_t b, int64_t e) {
    // 波段在外层、像元在内层：每个波段是一段连续内存，对缓存/预取友好；每个像元仍然是从 0.f 开始
    // 按波段 0..bands-1 顺序累加（np.sum(raw, axis=0) 的口径），逐位结果不变。
    std::vector<float> sum(static_cast<size_t>(e - b), 0.f);
    for (int b2 = 0; b2 < bands; ++b2) {
      const float* row = raw + static_cast<int64_t>(b2) * npix;
      for (int64_t p = b; p < e; ++p) sum[p - b] += row[p];
    }
    int64_t local = 0;
    for (int64_t p = b; p < e; ++p) {
      const float denom = nir[p] + red[p];
      const float ndvi = std::fabs(denom) > 1e-12f ? (nir[p] - red[p]) / denom : 0.f;
      mask[p] = (ndvi > ndvi_threshold) && (sum[p - b] > nodata_epsilon);
      local += mask[p];
    }
    if (out_valid_count) count += local;  // 每个线程只在自己的区间结束时加一次，不是逐像元原子操作
  });
  if (out_valid_count) *out_valid_count = count.load();
}

void hsi_band_stats(const float* raw, int bands, int64_t npix, float* mean, float* stdv, int workers) {
  const float n = static_cast<float>(npix);
  parallel_for(bands, workers, [&](int64_t b0, int64_t b1) {
    std::vector<float> sq(npix);
    for (int64_t b = b0; b < b1; ++b) {
      const float* a = raw + b * npix;
      const float m = np_sum(a, npix) / n;  // raw.mean(axis=(1,2))：float32 求和后 float32 除
      for (int64_t i = 0; i < npix; ++i) {
        const float d = a[i] - m;
        sq[i] = d * d;                            // (raw - mean) 的平方，先落成 float32 再求和
      }
      const float var = np_sum(sq.data(), npix) / n;
      mean[b] = m;
      stdv[b] = std::sqrt(var) + 1e-8f;
    }
  });
}
