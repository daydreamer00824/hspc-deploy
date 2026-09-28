// HSI 前处理的 CPU 部分：NDVI 有效像元掩膜、按波段均值/标准差。
// 与 scripts/preprocess.py 的 valid_vegetation_mask / standardize_hsi 逐位相同：
// 归约顺序复现 NumPy（8192 元素分块 + 块内 float32 pairwise 求和），逐元素运算全部是 IEEE float32。
#pragma once
#include <cstdint>

// raw: (bands, rows*cols) BSQ；mask: rows*cols 个 0/1。out_valid_count 非空时顺带统计有效像元数
// （在已有的并行遍历里按线程累加，不再需要调用方事后再扫一遍 mask）。
void hsi_valid_mask(const float* raw, int bands, int64_t npix, int red_index, int nir_index, float ndvi_threshold,
                    float nodata_epsilon, uint8_t* mask, int workers, int64_t* out_valid_count = nullptr);

// 每个波段在全部像元上的均值与标准差（std 已加 1e-8）。
void hsi_band_stats(const float* raw, int bands, int64_t npix, float* mean, float* stdv, int workers);
