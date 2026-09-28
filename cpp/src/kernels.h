// 自定义 CUDA kernel：标准化 + patch 提取融合、特征归一化、融合窗口匹配。
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

constexpr int kFeatDim = 1024;
constexpr int kVariants = 2;  // 匹配规则的 variant 数（与私有的 matching_rule.cuh 里的 kNumVariants 一致，kernels.cu 里 static_assert）

// patches[i][b][dy][dx] = (cube[b][vr[i]+dy-1][vc[i]+dx-1] - mean[b]) / std[b]，越界补 0。
// 等价于 Python：整幅标准化后四周补 0，再按 (row, col) 取 3x3 窗口；逐元素 IEEE float32 运算，结果逐位相同。
void launch_normalize_gather(const float* cube, int bands, int rows, int cols, const float* mean, const float* stdv,
                             const int32_t* vr, const int32_t* vc, int n, float* patches, cudaStream_t s);

// out[i][:] = y[i][:] / (||y[i][:]||_2 + 1e-12)；y 为 half 或 float 的 (n, 1024) 特征
void launch_normalize_features(const void* y, bool y_is_half, int n, float* out, cudaStream_t s);

// 融合窗口匹配：每个参考点一个 block，直接按 lut（rows*cols → 特征行号，无效像元为 -1）从归一化的 HSI 特征表里读窗口，
// 不拼 rows*cols*1024 的整幅特征网格。输出每个 variant 的最佳行列与该点的 cosine，无候选时行列为 -1。
void launch_match(const float* hsi_feat_norm, const int32_t* lut, const float* pc_feat_norm, const int32_t* ref_rows,
                  const int32_t* ref_cols, int n_points, int rows, int cols, int radius, int32_t* out_rows,
                  int32_t* out_cols, float* out_cos, cudaStream_t s);
