// 整景部署链路（C++）：原始 HSI + LAS → 跨模态匹配结果，不依赖 Python / torch。
//   读 HSI → NDVI 掩膜 + 按波段均值/标准差 → 上传立方体，融合 kernel 标准化并提取 patch → TensorRT HSI（异步）
//   → LAS 段（CPU，与 GPU 上的 HSI 推理重叠）→ TensorRT PC（第二个 stream）→ 特征归一化 + 融合窗口匹配 → 取回结果
// 对应 scripts/deploy_jetson.py 的最终配置 E，匹配结果与之逐位对照（scripts/cpp_guardrail.py）。
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cuda_utils.h"
#include "kernels.h"
#include "las_pipeline.h"
#include "trt_engine.h"

struct PipelineOptions {
  int cpu_workers = 1;                // HSI 掩膜/统计量的线程数（<=0 用全部核）
  int proj_workers = 1;               // LAS 投影线程数
  int query_workers = -1;             // kNN 查询线程数（-1 全部核）
  bool parallel_tree_build = true;    // kd 树构建放后台线程
  bool bulk_read = false;             // HSI 文件整块读（否则 GDAL RasterIO 逐波段逐行读）
  bool cache_projector = false;       // 跨景复用 PROJ 变换对象
  bool overlap_las_hsi_read = false;  // LAS 段（读文件、建树、投影）与 HSI 读取/前处理同时开始
};

struct MatchResult {
  int n_valid = 0, n_points = 0;
  std::vector<int32_t> rows[kVariants], cols[kVariants];
  std::vector<float> cosine[kVariants];
};

// 只在护栏检查时打开：把中间结果拷回 host，与 Python 链路逐数组比较
struct DebugOut {
  std::vector<float> patches;   // (n_valid, 342, 3, 3)
  std::vector<float> hsi_feat;  // (n_valid, 1024)，float32（半精度输出已展开）
  std::vector<float> pc_feat;   // (n_points, 1024)
  std::vector<float> offsets;   // (n_points, 15, 3)
  std::vector<uint8_t> mask;
  std::vector<int32_t> ref_rows, ref_cols;
  int rows = 0, cols = 0;
};

class DeployPipeline {
 public:
  DeployPipeline(const std::string& hsi_plan, const std::string& pc_plan, int max_batch);
  // seg_ms 得到各段耗时（毫秒）；开启 overlap_las_hsi_read 时 LAS 各段与 HSI 各段在时间上重叠，是各自的墙钟跨度。
  MatchResult run(const std::string& hsi_path, const std::string& las_path, const PipelineOptions& opt,
                  std::map<std::string, double>& seg_ms, DebugOut* dbg = nullptr);
  bool hsi_out_half() const { return hsi_.out_dtype() == nvinfer1::DataType::kHALF; }

 private:
  template <typename B>  // 只增不减的设备缓冲：够用就不重新分配

  static void ensure_dev(B& b, size_t n) { if (b.size() < n) b.alloc(n + n / 8); }

  TrtEngine hsi_, pc_;
  Stream sa_, sb_;
  Event ev_pc_, ev_hsi0_, ev_hsi1_, ev_pc0_, ev_m0_, ev_m1_;  // ev_pc_ 兼作 PC 推理结束；其余用于统计 GPU 各段耗时
  PinnedBuffer<float> h_raw_, h_stats_, h_off_;
  PinnedBuffer<int32_t> h_vrvc_, h_lut_, h_ref_, h_out_rows_, h_out_cols_;
  PinnedBuffer<float> h_out_cos_;
  PinnedBuffer<uint8_t> mask_;  // 有效像元掩膜，常驻缓冲区复用，避免整景大小的分配与清零
  DeviceBuffer<float> d_cube_, d_stats_, d_patches_, d_hsi_norm_, d_pc_in_, d_pc_norm_, d_out_cos_;
  DeviceBuffer<char> d_y_, d_pcy_;
  DeviceBuffer<int32_t> d_vrvc_, d_lut_, d_ref_, d_out_rows_, d_out_cols_;
};
