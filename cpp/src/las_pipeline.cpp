#include "las_pipeline.h"

#include <proj.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "ckdtree_decl.h"
#include "common_util.h"
#include "np_random.h"
#include "threading.h"

namespace {
template <typename T>
T rd(const std::vector<char>& b, size_t off) {
  T v;
  std::memcpy(&v, b.data() + off, sizeof(T));
  return v;
}
}  // namespace

std::vector<double> read_las_xyz(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open LAS: " + path);
  std::vector<char> b(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(b.data(), static_cast<std::streamsize>(b.size()));
  if (b.size() < 227 || std::memcmp(b.data(), "LASF", 4) != 0) throw std::runtime_error("not a LAS file: " + path);
  const uint8_t vmajor = rd<uint8_t>(b, 24), vminor = rd<uint8_t>(b, 25);
  const uint32_t data_off = rd<uint32_t>(b, 96);
  const uint8_t fmt = rd<uint8_t>(b, 104);
  const uint16_t reclen = rd<uint16_t>(b, 105);
  if (fmt & 0xC0) throw std::runtime_error("compressed LAZ is not supported: " + path);
  if (reclen < 12) throw std::runtime_error("LAS point record too short (<12 bytes): " + path);
  uint64_t npts = rd<uint32_t>(b, 107);
  if (vmajor == 1 && vminor >= 4) {
    if (b.size() < 255) throw std::runtime_error("truncated LAS 1.4 header: " + path);  // 1.4 的点数在偏移 247，占 8 字节
    npts = rd<uint64_t>(b, 247);
  }
  double scale[3], offset[3];
  std::memcpy(scale, b.data() + 131, sizeof scale);
  std::memcpy(offset, b.data() + 155, sizeof offset);
  if (data_off > b.size()) throw std::runtime_error("invalid LAS header (point data offset beyond EOF): " + path);
  // 用除法而不是 data_off + npts*reclen 直接比较：npts 来自文件、reclen 最大 65535，乘积可能溢出 64 位
  if (npts > (b.size() - data_off) / reclen) throw std::runtime_error("truncated LAS: " + path);
  std::vector<double> xyz(npts * 3);
  for (uint64_t i = 0; i < npts; ++i) {
    const char* rec = b.data() + data_off + i * reclen;
    int32_t v[3];
    std::memcpy(v, rec, 12);
    for (int d = 0; d < 3; ++d) {
      const double prod = static_cast<double>(v[d]) * scale[d];  // 两步运算，不做 FMA（编译选项 -ffp-contract=off）
      xyz[i * 3 + d] = prod + offset[d];
    }
  }
  return xyz;
}

// ---------------------------------------------------------------- KdTree
struct KdTree::Impl {
  ckdtree t{};
  std::vector<ckdtreenode> buffer;
  std::vector<intptr_t> indices;
  double maxes[3], mins[3];
};

KdTree::KdTree(const double* xyz, int64_t n) : p_(new Impl) {
  Impl& s = *p_;
  s.indices.resize(n);
  for (int64_t i = 0; i < n; ++i) s.indices[i] = i;
  for (int d = 0; d < 3; ++d) {
    s.maxes[d] = s.mins[d] = xyz[d];
    for (int64_t i = 1; i < n; ++i) {
      s.maxes[d] = std::max(s.maxes[d], xyz[i * 3 + d]);
      s.mins[d] = std::min(s.mins[d], xyz[i * 3 + d]);
    }
  }
  s.t.tree_buffer = &s.buffer;
  s.t.raw_data = const_cast<double*>(xyz);
  s.t.n = n;
  s.t.m = 3;
  s.t.leafsize = 16;
  s.t.raw_maxes = s.maxes;
  s.t.raw_mins = s.mins;
  s.t.raw_indices = s.indices.data();
  s.t.raw_boxsize_data = nullptr;
  // balanced_tree=False → _median=0（滑动中点规则），compact_nodes=True → _compact=1
  double mx[3] = {s.maxes[0], s.maxes[1], s.maxes[2]}, mn[3] = {s.mins[0], s.mins[1], s.mins[2]};
  build_ckdtree(&s.t, 0, n, mx, mn, /*_median=*/0, /*_compact=*/1);
  s.t.ctree = s.buffer.data();
  s.t.size = static_cast<intptr_t>(s.buffer.size());
  // 建树过程中 vector 可能扩容，节点里的 less/greater 指针要按下标重新指一遍（cKDTree._post_init 的做法）
  for (auto& node : s.buffer) {
    if (node.split_dim == -1) { node.less = nullptr; node.greater = nullptr; }
    else { node.less = s.t.ctree + node._less; node.greater = s.t.ctree + node._greater; }
  }
}
KdTree::~KdTree() = default;

void KdTree::query(const double* q, int64_t nq, int k, int workers, int64_t* out_idx) const {
  const intptr_t kk = k;
  std::vector<intptr_t> ranks(k);  // 要返回第 1..k 近的邻居（cKDTree.query(k=int) 传入的就是 arange(1, k+1)）
  for (int i = 0; i < k; ++i) ranks[i] = i + 1;
  parallel_for(nq, workers, [&](int64_t b, int64_t e) {
    std::vector<double> dd((e - b) * k);
    std::vector<intptr_t> ii((e - b) * k);
    query_knn(&p_->t, dd.data(), ii.data(), q + b * 3, e - b, ranks.data(), kk, kk, 0.0, 2.0, INFINITY);
    for (int64_t i = 0; i < (e - b) * k; ++i) out_idx[b * k + i] = ii[i];
  });
}

// ---------------------------------------------------------------- PROJ
namespace {
struct Projector {
  PJ_CONTEXT* ctx = nullptr;
  PJ* pj = nullptr;
  Projector(const std::string& src, const std::string& dst_wkt, const std::string& expected_target) {
    ctx = proj_context_create();
    // 中途任何一步抛异常都要先把已经创建的 PJ/上下文清理掉再重新抛出：构造函数没走完，
    // 析构函数不会被调用，ctx/s/t/e/raw 这些对象不会自己释放。
    PJ* s = nullptr;
    PJ* t = nullptr;
    PJ* e = nullptr;
    PJ* raw = nullptr;
    try {
      s = proj_create(ctx, src.c_str());
      t = proj_create(ctx, dst_wkt.c_str());
      e = proj_create(ctx, expected_target.c_str());
      if (!s || !t || !e) throw std::runtime_error("PROJ: cannot create CRS");
      if (!proj_is_equivalent_to_with_ctx(ctx, t, e, PJ_COMP_EQUIVALENT)) throw std::runtime_error("HSI CRS contract mismatch");
      raw = proj_create_crs_to_crs_from_pj(ctx, s, t, nullptr, nullptr);
      if (!raw) throw std::runtime_error("PROJ: cannot create transformation");
      pj = proj_normalize_for_visualization(ctx, raw);  // always_xy=True
      if (!pj) throw std::runtime_error("PROJ: normalize failed");
    } catch (...) {
      proj_destroy(raw); proj_destroy(s); proj_destroy(t); proj_destroy(e);
      if (ctx) proj_context_destroy(ctx);
      ctx = nullptr;
      throw;
    }
    proj_destroy(raw); proj_destroy(s); proj_destroy(t); proj_destroy(e);
  }
  ~Projector() {
    if (pj) proj_destroy(pj);
    if (ctx) proj_context_destroy(ctx);
  }
  Projector(const Projector&) = delete;
  Projector& operator=(const Projector&) = delete;
  void transform(double* x, double* y, size_t n) const {
    proj_trans_generic(pj, PJ_FWD, x, sizeof(double), n, y, sizeof(double), n, nullptr, 0, 0, nullptr, 0, 0);
  }
};
}  // namespace

// ---------------------------------------------------------------- 整个 LAS 段
LasResult run_las_pipeline(const std::string& las_path, const LasContract& c, const std::string& hsi_wkt, const double gt[6],
                           const std::function<const uint8_t*()>& wait_mask, int rows, int cols, const LasOptions& opt) {
  LasResult r;
  auto& t = r.seg_sec;
  double t0 = now_sec();
  std::vector<double> xyz = read_las_xyz(las_path);
  const int64_t N = static_cast<int64_t>(xyz.size() / 3);
  t["laspy_read_io"] = now_sec() - t0;
  if (N < c.k_neighbors + 1) throw std::runtime_error("too few points");
  // cur[64] 是下面 kNN 邻域拼接用的定长栈数组，最坏情况（查询到的 QK=k_neighbors+1 个近邻里一个都不是
  // 采样点自己）要装满 k_neighbors+1 个下标，超过 64 会栈溢出、覆盖别的局部变量。
  if (c.k_neighbors + 1 > 64) throw std::runtime_error("k_neighbors too large for fixed-size neighbor buffer (max 63)");

  std::future<std::unique_ptr<KdTree>> tree_future;
  if (opt.parallel_tree_build)
    tree_future = std::async(std::launch::async, [&] { return std::make_unique<KdTree>(xyz.data(), N); });

  t0 = now_sec();
  const int pw = opt.proj_workers > 0 ? opt.proj_workers : std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  // 每个线程各自一个 PJ 上下文（PROJ 对象不是线程安全的）；构造本身也并行。cache_projector 时放进进程级缓存，下一景直接复用
  static std::mutex cache_mu;
  static std::map<std::string, std::vector<std::shared_ptr<Projector>>> cache;
  std::vector<std::shared_ptr<Projector>> proj(pw);
  {
    const std::string key = c.point_cloud_crs + "|" + c.expected_target + "|" + hsi_wkt;
    std::lock_guard<std::mutex> lk(cache_mu);  // 只保护缓存表；构造在锁内并行完成，不同景不会同时进入这里
    auto& slot = cache[key];
    if (opt.cache_projector) {
      if (static_cast<int>(slot.size()) < pw) slot.resize(pw);
      for (int i = 0; i < pw; ++i) proj[i] = slot[i];
    }
    std::vector<std::thread> th;
    std::vector<std::exception_ptr> errs(pw);
    for (int i = 0; i < pw; ++i)
      if (!proj[i]) th.emplace_back([&, i] {
        // Projector 的构造函数会在 CRS 不合法/契约不匹配时抛异常，未捕获会直接 std::terminate，
        // 所以这里先捕获、再转交主线程重新抛出。
        try { proj[i] = std::make_shared<Projector>(c.point_cloud_crs, hsi_wkt, c.expected_target); }
        catch (...) { errs[i] = std::current_exception(); }
      });
    for (auto& x : th) x.join();
    for (auto& e : errs) if (e) std::rethrow_exception(e);
    if (opt.cache_projector)
      for (int i = 0; i < pw; ++i) slot[i] = proj[i];
  }
  t["crs_transformer_init"] = now_sec() - t0;

  t0 = now_sec();
  std::vector<double> hx(N), hy(N);
  for (int64_t i = 0; i < N; ++i) { hx[i] = xyz[i * 3]; hy[i] = xyz[i * 3 + 1]; }
  // 按 pw 个块分给对应下标的 proj[w]（每个线程独立的 PJ 上下文）；分块规则与 threading.h 的 parallel_for 相同，
  // 这里要用 parallel_chunks 是因为需要拿到 worker 下标去索引 proj[w]，parallel_for 不暴露这个下标。
  parallel_chunks(N, pw, [&](int w, int64_t b, int64_t e) { proj[w]->transform(hx.data() + b, hy.data() + b, e - b); });
  // map_to_canonical_pixel：逆矩阵乘 (x-gt0, y-gt3)，再向下取整
  const double a = gt[1], b_ = gt[2], cc_ = gt[4], d = gt[5];
  const double det = a * d - b_ * cc_;
  if (!(std::fabs(det) >= 1e-18)) throw std::runtime_error("Non-invertible GeoTransform");
  const double i00 = d / det, i01 = -b_ / det, i10 = -cc_ / det, i11 = a / det;
  std::vector<int64_t> rr(N), cc(N);
  for (int64_t i = 0; i < N; ++i) {
    const double dx = hx[i] - gt[0], dy = hy[i] - gt[3];
    const double col = i00 * dx + i01 * dy, row = i10 * dx + i11 * dy;
    if (!std::isfinite(row) || !std::isfinite(col)) throw std::runtime_error("Non-finite fractional pixel coordinate");
    rr[i] = static_cast<int64_t>(std::floor(row));
    cc[i] = static_cast<int64_t>(std::floor(col));
  }
  t["projection"] = now_sec() - t0;

  t0 = now_sec();
  const uint8_t* valid_mask = wait_mask();
  std::vector<int64_t> eligible;
  eligible.reserve(N);
  for (int64_t i = 0; i < N; ++i)
    if (rr[i] >= 0 && rr[i] < rows && cc[i] >= 0 && cc[i] < cols && valid_mask[rr[i] * cols + cc[i]]) eligible.push_back(i);
  t["mask_filter"] = now_sec() - t0;

  t0 = now_sec();
  if (static_cast<int64_t>(eligible.size()) > c.samples) eligible = np_choice_sorted(c.seed, eligible, c.samples);
  t["rng_sample"] = now_sec() - t0;

  t0 = now_sec();
  std::unique_ptr<KdTree> tree;
  if (opt.parallel_tree_build) {
    tree = tree_future.get();
    t["ckdtree_build_wait"] = now_sec() - t0;
  } else {
    tree = std::make_unique<KdTree>(xyz.data(), N);
    t["ckdtree_build"] = now_sec() - t0;
  }

  t0 = now_sec();
  const int K = c.k_neighbors, QK = std::min<int64_t>(K + 1, N);
  const int64_t n = static_cast<int64_t>(eligible.size());
  std::vector<double> q(n * 3);
  for (int64_t j = 0; j < n; ++j) std::memcpy(&q[j * 3], &xyz[eligible[j] * 3], 3 * sizeof(double));
  std::vector<int64_t> nb(n * QK);
  tree->query(q.data(), n, static_cast<int>(QK), opt.query_workers, nb.data());
  r.offsets.resize(n * K * 3);
  for (int64_t j = 0; j < n; ++j) {
    const int64_t pi = eligible[j];
    int64_t cur[64];
    int m = 0;
    for (int64_t k = 0; k < QK; ++k)
      if (nb[j * QK + k] != pi) cur[m++] = nb[j * QK + k];
    if (m == 0) cur[m++] = pi;
    while (m < K) { cur[m] = cur[m - 1]; ++m; }  // np.pad(mode="edge")
    for (int k = 0; k < K; ++k)
      for (int dd = 0; dd < 3; ++dd)
        r.offsets[(j * K + k) * 3 + dd] = static_cast<float>(xyz[cur[k] * 3 + dd] - xyz[pi * 3 + dd]);
  }
  t["ckdtree_query_offsets"] = now_sec() - t0;

  r.ref_rows.resize(n);
  r.ref_cols.resize(n);
  for (int64_t j = 0; j < n; ++j) { r.ref_rows[j] = rr[eligible[j]]; r.ref_cols[j] = cc[eligible[j]]; }
  r.eligible = std::move(eligible);
  if (opt.keep_debug) {
    r.xyz = std::move(xyz);
    r.hsi_x = std::move(hx);
    r.hsi_y = std::move(hy);
    r.full_rows = std::move(rr);
    r.full_cols = std::move(cc);
  }
  return r;
}
