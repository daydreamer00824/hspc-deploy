// hspc_deploy：Jetson 上的 C++ 整景部署程序。从原始 HSI + LAS 文件到 1000 个点的匹配行列。
//   hspc_deploy --scene 24data/10.6/1 [--config K4] [--out result.json] [--dump-dir D]   （不传 --config 时默认最终配置 K4）
// 计时口径与 scripts/deploy_jetson.py 一致：进程内首次整景单独报告（冷启动），之后 warmup 2 次 + 重复 10 次取各段中位数；
// 页缓存不清（文件 IO 是"文件缓存热"的数字）。
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common_util.h"
#include "deploy_pipeline.h"

static long vm_hwm_kb() {
  std::ifstream f("/proc/self/status");
  for (std::string l; std::getline(f, l);)
    if (l.rfind("VmHWM:", 0) == 0) return std::stol(l.substr(6));
  return -1;
}
template <typename T>
static std::string arr(const std::vector<T>& v) {
  std::ostringstream o;
  o << "[";
  for (size_t i = 0; i < v.size(); ++i) o << (i ? ", " : "") << v[i];
  o << "]";
  return o.str();
}

static bool r_equal(const MatchResult& a, const MatchResult& b) {
  for (int v = 0; v < kVariants; ++v)
    if (a.rows[v] != b.rows[v] || a.cols[v] != b.cols[v]) return false;
  return true;
}

// K 系列配置：累加对比，每一步只改一处（见 docs/jetson.md）。表按顺序描述"从这一步起追加的改动"，
// K0 是 Python 版 E 的调度原样移植（单线程前处理/投影、GDAL 逐行读、每景重建 PROJ 变换、LAS 段在 HSI 读完之后）。
static const std::vector<std::pair<const char*, void (*)(PipelineOptions&)>> kConfigChain = {
    {"K0", [](PipelineOptions&) {}},
    {"K1", [](PipelineOptions& o) { o.cpu_workers = 0; }},              // + HSI 掩膜/统计量多线程
    {"K2", [](PipelineOptions& o) { o.proj_workers = 0; }},             // + 投影多线程
    {"K3", [](PipelineOptions& o) { o.bulk_read = true; }},             // + HSI 文件整块读
    {"K4", [](PipelineOptions& o) { o.cache_projector = true; }},       // + 跨景复用 PROJ 变换对象
    {"K5", [](PipelineOptions& o) { o.overlap_las_hsi_read = true; }},  // + LAS 段与 HSI 读取/前处理同时开始
};

static bool apply_config(const std::string& c, PipelineOptions& o) {
  o = PipelineOptions{};
  for (auto& [name, mutate] : kConfigChain) {
    mutate(o);
    if (c == name) return true;
  }
  return false;
}

int main(int argc, char** argv) {
  std::string scene = "24data/10.6/1", config = "K4", out, dump_dir;  // K4 是最终配置；K5 实测更慢，未采纳（见 docs/jetson.md）
  std::string hsi_engine = "engines/hsi_fp16_scene1024.plan", pc_engine = "engines/pc_fp16_scene1024.plan";
  int warmup = 2, repeat = 10, max_batch = 1024;
  bool cold_only = false;  // 只跑进程内首次整景（冷启动测量用）
  double seconds = 0;  // >0：整景连续跑够这么多秒（能效测量用），代替固定次数
  const char* root_env = std::getenv("HSPC_DATA_ROOT");
  std::string data_root = root_env ? root_env : "data";
  try {
    ArgParser ap(argc, argv);
    for (std::string k; ap.next(k);) {
      if (k == "--scene") scene = ap.value();
      else if (k == "--config") config = ap.value();
      else if (k == "--out") out = ap.value();
      else if (k == "--dump-dir") dump_dir = ap.value();
      else if (k == "--data-root") data_root = ap.value();
      else if (k == "--hsi-engine") hsi_engine = ap.value();
      else if (k == "--pc-engine") pc_engine = ap.value();
      else if (k == "--warmup") warmup = parse_int(ap.value());
      else if (k == "--repeat") repeat = parse_int(ap.value());
      else if (k == "--seconds") seconds = parse_double(ap.value());
      else if (k == "--cold-only") cold_only = true;
      else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); return kExitBadArgs; }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "argument error: %s\n", e.what());
    return kExitBadArgs;
  }
  // 计时循环至少要跑一次：median/输出都依赖至少一个结果
  if (warmup < 0 || repeat < 1 || seconds < 0) {
    std::fprintf(stderr, "argument error: need --warmup >= 0, --repeat >= 1, --seconds >= 0\n");
    return kExitBadArgs;
  }
  PipelineOptions opt;
  if (!apply_config(config, opt)) { std::fprintf(stderr, "unknown config %s (K0..K5)\n", config.c_str()); return kExitBadArgs; }
  const auto s1 = scene.find('/'), s2 = scene.find('/', s1 + 1);
  const std::string batch = scene.substr(0, s1), date = scene.substr(s1 + 1, s2 - s1 - 1), sid = scene.substr(s2 + 1);
  const std::string hsi_path = data_root + "/hsi_spatial_spectral_resampled_common_342/" + batch + "/hsi/" + date + "/" + sid + "_spec342.dat";
  const std::string las_path = data_root + "/lai_icp_registered_resampled_hsi/" + batch + "/" + date + "/rice_las/" + sid + "_rice_icp.las";

  try {
    double t0 = now_sec();
    DeployPipeline pipe(hsi_engine, pc_engine, max_batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double engine_load = now_sec() - t0;
    std::map<std::string, double> seg;

    if (!dump_dir.empty()) {  // 护栏模式：跑一次，导出中间结果
      DebugOut dbg;
      MatchResult r = pipe.run(hsi_path, las_path, opt, seg, &dbg);
      dump(dump_dir, "patches.f32", dbg.patches);
      dump(dump_dir, "hsi_feat.f32", dbg.hsi_feat);
      dump(dump_dir, "pc_feat.f32", dbg.pc_feat);
      dump(dump_dir, "offsets.f32", dbg.offsets);
      dump(dump_dir, "mask.u8", dbg.mask);
      dump(dump_dir, "ref_rows.i32", dbg.ref_rows);
      dump(dump_dir, "ref_cols.i32", dbg.ref_cols);
      for (int v = 0; v < kVariants; ++v) {
        dump(dump_dir, "match_rows_" + std::to_string(v) + ".i32", r.rows[v]);
        dump(dump_dir, "match_cols_" + std::to_string(v) + ".i32", r.cols[v]);
        dump(dump_dir, "match_cos_" + std::to_string(v) + ".f32", r.cosine[v]);
      }
      {
        std::ostringstream meta;
        meta << r.n_valid << " " << r.n_points << " " << dbg.rows << " " << dbg.cols << "\n";
        write_text_file(dump_dir + "/meta.txt", meta.str());
      }
      std::fprintf(stderr, "dumped to %s (n_valid=%d n_points=%d)\n", dump_dir.c_str(), r.n_valid, r.n_points);
      return 0;
    }

    t0 = now_sec();
    MatchResult cold = pipe.run(hsi_path, las_path, opt, seg, nullptr);
    const double cold_total = now_sec() - t0;
    std::map<std::string, double> cold_seg = seg;
    if (cold_only) {
      std::ostringstream cj;
      cj << "{\"scene\": \"" << scene << "\", \"config\": \"" << config << "\", \"engine_load_sec\": " << engine_load << ", \"cold_start\": {\"total_sec\": "
         << cold_total << "}, \"peak_rss_kb\": " << vm_hwm_kb() << "}\n";
      if (out.empty()) std::fputs(cj.str().c_str(), stdout); else write_text_file(out, cj.str());
      return 0;
    }
    for (int i = 0; i < warmup; ++i) pipe.run(hsi_path, las_path, opt, seg, nullptr);
    std::vector<double> totals;
    std::map<std::string, std::vector<double>> segs;
    // 只保留"第一次热身后结果"作参照和"最后一次结果"用于输出，逐次比较、不攒下整个数组——这样 --seconds 长时间模式
    // （能效测量，可能几十次）也能对"每一次"都做确定性检查，而不是只在结尾比较剩下的一两次。
    MatchResult ref, last_result;
    bool have_ref = false, deterministic = true;
    const double wall0 = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    const double loop0 = now_sec();
    // i == 0：--seconds 很小时也至少计时一次（否则后面对空数组取中位数）
    for (int i = 0; i == 0 || (seconds > 0 ? now_sec() - loop0 < seconds : i < repeat); ++i) {
      t0 = now_sec();
      MatchResult r = pipe.run(hsi_path, las_path, opt, seg, nullptr);
      totals.push_back((now_sec() - t0) * 1000.0);
      for (auto& kv : seg) segs[kv.first].push_back(kv.second);
      if (!have_ref) { ref = r; have_ref = true; deterministic = r_equal(cold, r); }
      else deterministic = deterministic && r_equal(ref, r);
      last_result = std::move(r);
    }
    const double wall1 = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    const int n_timed = static_cast<int>(totals.size());

    double io_ms = 0;
    std::ostringstream js;
    js << "{\n  \"scene\": \"" << scene << "\", \"config\": \"" << config << "\", \"warmup\": " << warmup << ", \"repeat\": " << repeat
       << ",\n  \"options\": {\"cpu_workers\": " << opt.cpu_workers << ", \"proj_workers\": " << opt.proj_workers
       << ", \"query_workers\": " << opt.query_workers << ", \"parallel_tree_build\": " << json_bool(opt.parallel_tree_build)
       << ", \"bulk_read\": " << json_bool(opt.bulk_read) << ", \"cache_projector\": " << json_bool(opt.cache_projector)
       << ", \"overlap_las_hsi_read\": " << json_bool(opt.overlap_las_hsi_read) << "},"
       << "\n  \"hsi_engine\": \"" << hsi_engine << "\", \"pc_engine\": \"" << pc_engine << "\", \"n_valid_pixels\": " << last_result.n_valid
       << ", \"n_points\": " << last_result.n_points << ", \"engine_load_sec\": " << engine_load << ",\n  \"cold_start\": {\"total_sec\": "
       << cold_total << ", \"segments_ms\": {";
    bool first = true;
    for (auto& kv : cold_seg) { js << (first ? "" : ", ") << "\"" << kv.first << "\": " << kv.second; first = false; }
    js << "}},\n  \"warm\": {\"n_timed\": " << n_timed << ", \"t_start_epoch\": " << std::fixed << wall0 << ", \"t_end_epoch\": " << wall1 << std::defaultfloat << ", \"total_ms_median\": " << median(totals) << ", \"total_ms_all\": " << arr(totals) << ", \"segments_ms_median\": {";
    first = true;
    for (auto& kv : segs) {
      const double m = median(kv.second);
      if (kv.first == "load_hsi_io" || kv.first == "laspy_read_io") io_ms += m;
      js << (first ? "" : ", ") << "\"" << kv.first << "\": " << m;
      first = false;
    }
    js << "}, \"io_ms_median_sum\": " << io_ms << ", \"total_without_io_ms_median\": " << median(totals) - io_ms << "},\n"
       << "  \"determinism_match_rowcols_identical_across_reps\": " << json_bool(deterministic)
       << ",\n  \"peak_rss_kb\": " << vm_hwm_kb() << ",\n  \"match_rowcols\": {";
    for (int v = 0; v < kVariants; ++v)
      js << (v ? ", " : "") << "\"variant" << v << "\": {\"rows\": " << arr(last_result.rows[v]) << ", \"cols\": " << arr(last_result.cols[v]) << "}";
    js << "}\n}\n";
    if (out.empty()) std::fputs(js.str().c_str(), stdout);
    else write_text_file(out, js.str());
    std::fprintf(stderr, "%s %s: total %.1f ms (median of %d), cold %.1f ms, deterministic=%d\n", scene.c_str(), config.c_str(), median(totals),
                 n_timed, cold_total * 1000, deterministic);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
