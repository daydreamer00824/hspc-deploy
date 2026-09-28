// 单样本延迟：batch 1/8，eager vs CUDA Graph。口径与 scripts/jetson_scene_opt.py 的 cmd_latency 一致：
//   event   = 只计 GPU 段（CUDA event 夹住 run）
//   wall    = 下发 + 同步
//   request = 页锁定内存 H2D 输入拷贝 + run + D2H 输出拷贝 + 同步（最接近真实请求）
// 输出 JSON；--dump-dir 时把每个 (batch, mode) 的输出按 float32 写盘，供与 Python 结果逐位比较。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common_util.h"
#include "cuda_utils.h"
#include "trt_engine.h"

struct Stats { double median, p95, p99; };

static Stats stats(const std::vector<double>& v) { return {percentile(v, 50), percentile(v, 95), percentile(v, 99)}; }

// row_elems：每条样本的元素数（dims 各维相乘），用来校验文件大小是整数条样本，不是被截断的数据。
static std::vector<float> read_f32(const std::string& path, size_t row_elems) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  const size_t bytes = static_cast<size_t>(f.tellg());
  if (bytes == 0) throw std::runtime_error("samples file is empty: " + path);
  if (bytes % (row_elems * sizeof(float)) != 0)
    throw std::runtime_error("samples file size (" + std::to_string(bytes) + " bytes) is not a whole number of rows ("
                              + std::to_string(row_elems) + " floats each): " + path);
  std::vector<float> v(bytes / sizeof(float));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
  if (!f) throw std::runtime_error("failed to read " + path);
  return v;
}

static std::vector<float> to_float(const void* p, size_t n, nvinfer1::DataType t) {
  std::vector<float> o(n);
  TrtEngine::expand_to_float(p, n, t, o.data());
  return o;
}

static std::string stat_json(const Stats& s) {
  char b[160];
  std::snprintf(b, sizeof b, "{\"median_ms\": %.6f, \"p95_ms\": %.6f, \"p99_ms\": %.6f}", s.median, s.p95, s.p99);
  return b;
}

// TrtEngine 的 max_batch：引擎构建 profile 的上限，batch 超过这个值就不是本工具要测的单请求延迟场景了。
constexpr int kMaxBatch = 64;

int main(int argc, char** argv) {
  std::string which, plan, samples, dump_dir, out_json;
  std::vector<int> batches{1, 8};
  int iters = 300;
  try {
    ArgParser ap(argc, argv);
    for (std::string k; ap.next(k);) {
      if (k == "--which") which = ap.value();
      else if (k == "--engine") plan = ap.value();
      else if (k == "--samples") samples = ap.value();
      else if (k == "--iters") iters = parse_int(ap.value());
      else if (k == "--dump-dir") dump_dir = ap.value();
      else if (k == "--out") out_json = ap.value();
      else if (k == "--batches") {
        batches.clear();
        std::stringstream ss(ap.value());
        for (std::string t; std::getline(ss, t, ',');) batches.push_back(parse_int(t));
      } else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); return kExitBadArgs; }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "argument error: %s\n", e.what());
    return kExitBadArgs;
  }
  if (which.empty() || plan.empty() || samples.empty()) {
    std::fprintf(stderr, "usage: hspc_latency --which pc|hsi --engine X.plan --samples s.f32 [--batches 1,8] [--iters 300] [--dump-dir D] [--out J]\n");
    return kExitBadArgs;
  }
  if (which != "pc" && which != "hsi") {
    std::fprintf(stderr, "argument error: --which must be pc or hsi, got %s\n", which.c_str());
    return kExitBadArgs;
  }
  if (iters < 1) {  // stats()/percentile() 需要至少一个样本
    std::fprintf(stderr, "argument error: --iters must be >= 1\n");
    return kExitBadArgs;
  }
  if (batches.empty()) {
    std::fprintf(stderr, "argument error: --batches must not be empty\n");
    return kExitBadArgs;
  }
  {
    std::vector<int> seen;
    for (int b : batches) {
      if (b < 1 || b > kMaxBatch) {
        std::fprintf(stderr, "argument error: batch %d out of range [1, %d]\n", b, kMaxBatch);
        return kExitBadArgs;
      }
      if (std::find(seen.begin(), seen.end(), b) != seen.end()) {
        std::fprintf(stderr, "argument error: batch %d listed more than once in --batches\n", b);
        return kExitBadArgs;
      }
      seen.push_back(b);
    }
  }
  try {
    const std::vector<int64_t> dims = which == "pc" ? std::vector<int64_t>{15, 3} : std::vector<int64_t>{342, 3, 3};
    size_t row_elems = 1;
    for (int64_t d : dims) row_elems *= static_cast<size_t>(d);
    const std::vector<float> host_samples = read_f32(samples, row_elems);
    const int max_b = *std::max_element(batches.begin(), batches.end());
    const size_t have_rows = host_samples.size() / row_elems, need_rows = static_cast<size_t>(max_b);
    if (have_rows < need_rows)
      throw std::runtime_error("samples file has " + std::to_string(have_rows) + " rows, need at least "
                                + std::to_string(need_rows) + " for the largest --batches value (" + samples + ")");
    const int warmup = 50;
    std::ostringstream js;
    js << "{\"warmup\": " << warmup << ", \"measure\": " << iters << ", \"engines\": {";
    bool first_b = true;
    for (int b : batches) {
      std::vector<std::vector<float>> outs;
      std::string entry;
      Stats modes_event[2], modes_wall[2], modes_req[2];
      const char* names[2] = {"eager", "graph"};
      for (int mi = 0; mi < 2; ++mi) {
        TrtEngine eng(plan, dims, kMaxBatch);
        if (eng.in_dtype() != nvinfer1::DataType::kFLOAT) throw std::runtime_error("engine input must be float32");
        Stream stream;
        const size_t in_n = static_cast<size_t>(b) * eng.in_row_elems(), out_n = static_cast<size_t>(b) * eng.out_dim();
        DeviceBuffer<float> d_in(in_n);
        DeviceBuffer<char> d_out(out_n * eng.out_elem_bytes());
        CUDA_CHECK(cudaMemsetAsync(d_in.get(), 0, in_n * 4, stream.get()));  // 非阻塞 stream 与默认流之间没有隐式同步，必须在同一个 stream 上清零
        PinnedBuffer<float> h_in(in_n);
        PinnedBuffer<char> h_out(out_n * eng.out_elem_bytes());
        std::memcpy(h_in.host(), host_samples.data(), in_n * 4);
        if (mi == 1) eng.capture_graph(d_in.get(), d_out.get(), b, stream.get());
        auto run = [&]() {
          if (mi == 1) eng.replay(stream.get());
          else eng.infer(d_in.get(), d_out.get(), b, stream.get());
        };
        auto request = [&]() {
          CUDA_CHECK(cudaMemcpyAsync(d_in.get(), h_in.host(), in_n * 4, cudaMemcpyHostToDevice, stream.get()));
          run();
          CUDA_CHECK(cudaMemcpyAsync(h_out.host(), d_out.get(), h_out.size(), cudaMemcpyDeviceToHost, stream.get()));
          stream.sync();
        };
        request();  // 取输出用于逐位比较
        outs.push_back(to_float(h_out.host(), out_n, eng.out_dtype()));
        if (!dump_dir.empty())
          dump(dump_dir, which + "_b" + std::to_string(b) + "_" + names[mi] + ".f32", outs.back());
        for (int i = 0; i < warmup; ++i) request();
        std::vector<std::pair<Event, Event>> ev(iters);
        for (auto& p : ev) { p.first.record(stream.get()); run(); p.second.record(stream.get()); }
        stream.sync();
        std::vector<double> event_ms, wall, req;
        for (auto& p : ev) event_ms.push_back(Event::elapsed_ms(p.first, p.second));
        for (int i = 0; i < iters; ++i) { double t0 = now_ms(); run(); stream.sync(); wall.push_back(now_ms() - t0); }
        for (int i = 0; i < iters; ++i) { double t0 = now_ms(); request(); req.push_back(now_ms() - t0); }
        modes_event[mi] = stats(event_ms); modes_wall[mi] = stats(wall); modes_req[mi] = stats(req);
      }
      const bool identical = outs[0] == outs[1];
      if (!first_b) js << ", ";
      first_b = false;
      js << "\"" << which << "_b" << b << "\": {";
      for (int mi = 0; mi < 2; ++mi)
        js << "\"" << names[mi] << "\": {\"event\": " << stat_json(modes_event[mi]) << ", \"wall\": " << stat_json(modes_wall[mi])
           << ", \"request\": " << stat_json(modes_req[mi]) << "}, ";
      char sp[200];
      std::snprintf(sp, sizeof sp, "\"speedup_median\": {\"event\": %.4f, \"wall\": %.4f, \"request\": %.4f}",
                    modes_event[0].median / modes_event[1].median, modes_wall[0].median / modes_wall[1].median,
                    modes_req[0].median / modes_req[1].median);
      js << "\"graph_output_identical_to_eager\": " << json_bool(identical) << ", " << sp << "}";
      std::fprintf(stderr, "%s_b%d eager(event/wall/req)=%.3f/%.3f/%.3f graph=%.3f/%.3f/%.3f identical=%d\n", which.c_str(), b,
                   modes_event[0].median, modes_wall[0].median, modes_req[0].median, modes_event[1].median,
                   modes_wall[1].median, modes_req[1].median, identical);
    }
    js << "}}\n";
    if (out_json.empty()) std::fputs(js.str().c_str(), stdout);
    else write_text_file(out_json, js.str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
