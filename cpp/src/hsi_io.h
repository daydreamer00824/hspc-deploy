// 高光谱影像读取（GDAL C API）：ENVI BSQ float32 → (bands, rows, cols)，可直接读进调用方给的缓冲（如映射内存）。
#pragma once
#include <cstdint>
#include <string>

struct HsiHeader {
  int bands = 0, rows = 0, cols = 0;
  double gt[6] = {0, 0, 0, 0, 0, 0};
  std::string wkt;
  // ENVI BSQ float32 小端、无头部偏移时可以整块读文件（bulk 快速路径）；否则只能走 GDAL RasterIO
  std::string data_file;
  bool bulk_ok = false;
  int64_t pixels() const { return static_cast<int64_t>(rows) * cols; }
  int64_t elems() const { return pixels() * bands; }
};

// 只读头部（尺寸、geotransform、投影 WKT）。
HsiHeader hsi_read_header(const std::string& path);
// 把整个立方体读进 dst（bands*rows*cols 个 float，BSQ 顺序）。
// bulk=true 且文件满足快速路径条件时，一次 pread 整块读入；否则用 GDAL RasterIO（逐波段逐行读，结果逐位相同，但慢得多）。
void hsi_read_into(const std::string& path, const HsiHeader& h, float* dst, bool bulk);
