#include "hsi_io.h"

#include <fcntl.h>
#include <gdal.h>
#include <cpl_string.h>
#include <unistd.h>

#include <mutex>
#include <stdexcept>

namespace {
void gdal_init() {
  static std::once_flag once;
  std::call_once(once, [] { GDALAllRegister(); });
}
struct Dataset {
  GDALDatasetH h;
  explicit Dataset(const std::string& p) {
    gdal_init();
    h = GDALOpen(p.c_str(), GA_ReadOnly);
    if (!h) throw std::runtime_error("Cannot open HSI: " + p);
  }
  ~Dataset() { GDALClose(h); }
};
}  // namespace

HsiHeader hsi_read_header(const std::string& path) {
  Dataset ds(path);
  HsiHeader h;
  h.cols = GDALGetRasterXSize(ds.h);
  h.rows = GDALGetRasterYSize(ds.h);
  h.bands = GDALGetRasterCount(ds.h);
  if (GDALGetGeoTransform(ds.h, h.gt) != CE_None) throw std::runtime_error("no geotransform: " + path);
  const char* wkt = GDALGetProjectionRef(ds.h);
  h.wkt = wkt ? wkt : "";
  // ENVI 头部（GDAL 的 "ENVI" 元数据域）：判断能不能整块读
  char** md = GDALGetMetadata(ds.h, "ENVI");
  auto item = [&](const char* k) { const char* v = CSLFetchNameValue(md, k); return v ? std::string(v) : std::string(); };
  char** files = GDALGetFileList(ds.h);
  h.bulk_ok = item("interleave") == "bsq" && item("data_type") == "4" && item("byte_order") == "0" &&
              item("header_offset") == "0" && files && files[0] && GDALGetDriverByName("ENVI") == GDALGetDatasetDriver(ds.h);
  if (h.bulk_ok) h.data_file = files[0];
  CSLDestroy(files);
  return h;
}

void hsi_read_into(const std::string& path, const HsiHeader& h, float* dst, bool bulk) {
  if (bulk && h.bulk_ok) {
    const int fd = ::open(h.data_file.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + h.data_file);
    const int64_t total = h.elems() * static_cast<int64_t>(sizeof(float));
    int64_t got = 0;
    while (got < total) {
      const ssize_t r = ::pread(fd, reinterpret_cast<char*>(dst) + got, static_cast<size_t>(total - got), got);
      if (r <= 0) { ::close(fd); throw std::runtime_error("short read: " + h.data_file); }
      got += r;
    }
    ::close(fd);
    return;
  }
  Dataset ds(path);
  const int64_t npix = h.pixels();
  // 像元间距 4 字节、行间距 cols*4、波段间距 rows*cols*4：与 GDAL ReadAsArray 得到的 (bands, rows, cols) 完全一致
  if (GDALDatasetRasterIO(ds.h, GF_Read, 0, 0, h.cols, h.rows, dst, h.cols, h.rows, GDT_Float32, h.bands, nullptr,
                          4, static_cast<int64_t>(h.cols) * 4, npix * 4) != CE_None)
    throw std::runtime_error("GDALDatasetRasterIO failed: " + path);
}
