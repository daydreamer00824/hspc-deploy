// LAS 段：读点云 → PROJ 投影到 HSI 坐标系 → 像元行列 → 过滤/采样 → k 近邻 → 邻域坐标偏移。
// 与 scripts/deploy_scene.py 的 run_las_pipeline 逐步对应，结果逐位相同。
#pragma once
#include <cstdint>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>

// 点云读取：LAS 1.0-1.3（未压缩），点格式 0-3 等以 int32 X/Y/Z 开头的记录，坐标 = X*scale + offset（两步 float64 运算，与 laspy 一致）
std::vector<double> read_las_xyz(const std::string& path);  // (N,3) 行优先 float64

// scipy.spatial.cKDTree(xyz, balanced_tree=False) 的等价实现（直接使用 SciPy 的 C++ 内核）
class KdTree {
 public:
  explicit KdTree(const double* xyz, int64_t n);  // 数据不拷贝，调用方保证生命周期
  ~KdTree();
  KdTree(const KdTree&) = delete;
  KdTree& operator=(const KdTree&) = delete;
  // 对 nq 个查询点各取 k 个最近邻的下标（按距离升序）；workers <= 0 用全部核
  void query(const double* q, int64_t nq, int k, int workers, int64_t* out_idx) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

struct LasContract {
  std::string point_cloud_crs;   // "EPSG:32651"
  std::string expected_target;   // "EPSG:4326"
  int k_neighbors = 15;
  int samples = 1000;
  uint64_t seed = 0;
};

struct LasResult {
  std::vector<float> offsets;        // (n, k, 3)
  std::vector<int64_t> eligible;     // 采样后的点下标（升序）
  std::vector<int64_t> ref_rows, ref_cols;
  std::vector<double> xyz, hsi_x, hsi_y;  // 调试导出用（--dump）；hsi_x/y 为投影后的坐标
  std::vector<int64_t> full_rows, full_cols;
  std::map<std::string, double> seg_sec;  // 分段耗时，键名与 Python 版一致
};

struct LasOptions {
  int query_workers = -1;        // kNN 查询线程数
  bool parallel_tree_build = true;  // kd 树构建放到后台线程，与投影并行
  int proj_workers = 1;          // 投影线程数（每个线程各建一个 PROJ 上下文；<=0 用全部核）
  bool cache_projector = false;  // 跨调用复用 PROJ 上下文与变换对象（同一进程处理多景时不必每景重建）
  bool keep_debug = false;       // 保留 xyz、投影坐标、全部点的行列
};

// wait_mask 在需要 HSI 有效像元掩膜（rows*cols 个 0/1）时调用，可以阻塞到掩膜算好；这样读文件、建树、投影可以和 HSI 的读取/前处理同时进行。
// gt/wkt 来自 HSI 头部。
LasResult run_las_pipeline(const std::string& las_path, const LasContract& c, const std::string& hsi_wkt, const double gt[6],
                           const std::function<const uint8_t*()>& wait_mask, int rows, int cols, const LasOptions& opt);
