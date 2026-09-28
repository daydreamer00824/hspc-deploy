// C2 护栏用：跑一遍 CPU 前处理（HSI 读取、掩膜、均值/标准差、LAS 段），把中间数组按二进制导出，
// 由 scripts/cpp_guardrail.py 与 Python 参考链路逐数组比较。
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "common_util.h"
#include "contract.h"
#include "hsi_io.h"
#include "hsi_prep.h"
#include "las_pipeline.h"

int main(int argc, char** argv) {
  std::string hsi, las, dir;
  int workers = 0, proj_workers = 1;
  bool bulk = false;
  try {
    ArgParser ap(argc, argv);
    for (std::string k; ap.next(k);) {
      if (k == "--hsi") hsi = ap.value();
      else if (k == "--las") las = ap.value();
      else if (k == "--dump-dir") dir = ap.value();
      else if (k == "--workers") workers = parse_int(ap.value());
      else if (k == "--proj-workers") proj_workers = parse_int(ap.value());
      else if (k == "--bulk") bulk = true;
      else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); return kExitBadArgs; }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "argument error: %s\n", e.what());
    return kExitBadArgs;
  }
  if (hsi.empty() || las.empty() || dir.empty()) {
    std::fprintf(stderr, "usage: hspc_prep_check --hsi X.dat --las Y.las --dump-dir D [--workers N] [--proj-workers N] [--bulk]\n");
    return kExitBadArgs;
  }
  try {
    HsiHeader h = hsi_read_header(hsi);
    if (h.bands != contract::kBands)
      throw std::runtime_error("HSI band count mismatch: expected " + std::to_string(contract::kBands) + ", got " + std::to_string(h.bands));
    std::vector<float> raw(h.elems());
    hsi_read_into(hsi, h, raw.data(), bulk);
    if (bulk && !h.bulk_ok) std::fprintf(stderr, "note: bulk read not applicable, used GDAL\n");
    std::vector<uint8_t> mask(h.pixels());
    hsi_valid_mask(raw.data(), h.bands, h.pixels(), contract::kRedIndex, contract::kNirIndex, contract::kNdviThreshold,
                  contract::kNodataEpsilon, mask.data(), workers);
    std::vector<float> mean(h.bands), stdv(h.bands);
    hsi_band_stats(raw.data(), h.bands, h.pixels(), mean.data(), stdv.data(), workers);

    LasContract c{contract::kPointCloudCrs, contract::kExpectedTargetCrs, contract::kNeighbors, contract::kSamplesPerScene, contract::kSeed};
    LasOptions o;
    o.keep_debug = true;
    o.proj_workers = proj_workers;
    LasResult r = run_las_pipeline(las, c, h.wkt, h.gt, [&] { return mask.data(); }, h.rows, h.cols, o);

    {
      std::ostringstream shape;
      shape << h.bands << " " << h.rows << " " << h.cols << "\n";
      write_text_file(dir + "/shape.txt", shape.str());
    }
    dump(dir, "raw.f32", raw);
    dump(dir, "mask.u8", mask);
    dump(dir, "mean.f32", mean);
    dump(dir, "std.f32", stdv);
    dump(dir, "xyz.f64", r.xyz);
    dump(dir, "hsi_x.f64", r.hsi_x);
    dump(dir, "hsi_y.f64", r.hsi_y);
    dump(dir, "full_rows.i64", r.full_rows);
    dump(dir, "full_cols.i64", r.full_cols);
    dump(dir, "eligible.i64", r.eligible);
    dump(dir, "ref_rows.i64", r.ref_rows);
    dump(dir, "ref_cols.i64", r.ref_cols);
    dump(dir, "offsets.f32", r.offsets);
    std::vector<double> gt(h.gt, h.gt + 6);
    dump(dir, "gt.f64", gt);
    write_text_file(dir + "/wkt.txt", h.wkt);
    for (auto& kv : r.seg_sec) std::fprintf(stderr, "%s %.1f ms\n", kv.first.c_str(), kv.second * 1000);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
