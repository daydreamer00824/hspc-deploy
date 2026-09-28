#include "deploy_pipeline.h"

#include <cstring>
#include <exception>
#include <future>
#include <stdexcept>
#include <thread>

#include "common_util.h"
#include "contract.h"
#include "hsi_io.h"
#include "hsi_prep.h"

using contract::kBands;
using contract::kNeighbors;
using contract::kNirIndex;
using contract::kNdviThreshold;
using contract::kNodataEpsilon;
using contract::kPatchSize;
using contract::kRedIndex;
using contract::kSamplesPerScene;
using contract::kSearchRadius;
using contract::kSeed;

DeployPipeline::DeployPipeline(const std::string& hsi_plan, const std::string& pc_plan, int max_batch)
    : hsi_(hsi_plan, {kBands, kPatchSize, kPatchSize}, max_batch), pc_(pc_plan, {kNeighbors, 3}, max_batch) {
  if (hsi_.in_dtype() != nvinfer1::DataType::kFLOAT || pc_.in_dtype() != nvinfer1::DataType::kFLOAT)
    throw std::runtime_error("engine inputs must be float32");
  for (auto* e : {&hsi_, &pc_})
    if (e->out_dtype() != nvinfer1::DataType::kFLOAT && e->out_dtype() != nvinfer1::DataType::kHALF)
      throw std::runtime_error("engine output must be float32 or float16");
  // 后面所有 kernel（normalize/match）和缓冲区分配都按 kFeatDim=1024 写死；换一个输出维度不同的引擎
  // 时如果不在这里挡住，会在 kernels.cu 里读写越界，且不一定马上崩溃。
  if (hsi_.out_dim() != kFeatDim || pc_.out_dim() != kFeatDim)
    throw std::runtime_error("engine output dim must be " + std::to_string(kFeatDim) + ", got hsi="
                              + std::to_string(hsi_.out_dim()) + " pc=" + std::to_string(pc_.out_dim()));
}

MatchResult DeployPipeline::run(const std::string& hsi_path, const std::string& las_path, const PipelineOptions& opt,
                                std::map<std::string, double>& seg_ms, DebugOut* dbg) {
  seg_ms.clear();
  double t0 = now_ms();
  auto mark = [&](const char* name) { const double t = now_ms(); seg_ms[name] = t - t0; t0 = t; };

  HsiHeader h = hsi_read_header(hsi_path);
  if (h.bands != kBands) throw std::runtime_error("Expected exactly 342 bands");
  const int64_t npix = h.pixels();

  // ---- LAS 段作为一个任务：opt.overlap_las_hsi_read 时在后台线程立刻开始（它只在过滤那一步才需要掩膜）
  ensure_dev(mask_, static_cast<size_t>(npix));
  std::promise<const uint8_t*> mask_promise;
  auto mask_future = mask_promise.get_future().share();
  LasContract las_contract{contract::kPointCloudCrs, contract::kExpectedTargetCrs, kNeighbors, kSamplesPerScene, kSeed};
  LasOptions lopt;
  lopt.query_workers = opt.query_workers;
  lopt.parallel_tree_build = opt.parallel_tree_build;
  lopt.proj_workers = opt.proj_workers;
  lopt.cache_projector = opt.cache_projector;
  lopt.keep_debug = false;
  LasResult las;
  std::exception_ptr las_err;
  auto las_job = [&] {
    try {
      las = run_las_pipeline(las_path, las_contract, h.wkt, h.gt, [&] { return mask_future.get(); }, h.rows, h.cols, lopt);
    } catch (...) { las_err = std::current_exception(); }
  };
  std::thread las_thread;
  if (opt.overlap_las_hsi_read) las_thread = std::thread(las_job);
  auto fail = [&](std::exception_ptr e) {  // 主线程出错：让等掩膜的 LAS 线程也退出，再抛出
    try { mask_promise.set_exception(e); } catch (...) {}
    if (las_thread.joinable()) las_thread.join();
    std::rethrow_exception(e);
  };

  try {
    ensure_dev(h_raw_, static_cast<size_t>(h.elems()));
    hsi_read_into(hsi_path, h, h_raw_.host(), opt.bulk_read);
    mark("load_hsi_io");

    int64_t n64 = 0;
    hsi_valid_mask(h_raw_.host(), kBands, npix, kRedIndex, kNirIndex, kNdviThreshold, kNodataEpsilon, mask_.host(), opt.cpu_workers, &n64);
    mask_promise.set_value(mask_.host());
    ensure_dev(h_stats_, 2 * kBands);
    hsi_band_stats(h_raw_.host(), kBands, npix, h_stats_.host(), h_stats_.host() + kBands, opt.cpu_workers);
    const int n = static_cast<int>(n64);
    if (n == 0) throw std::runtime_error("no valid pixels");
    ensure_dev(h_vrvc_, 2u * n);
    ensure_dev(h_lut_, static_cast<size_t>(npix));
    {
      int32_t* vr = h_vrvc_.host();
      int32_t* vc = vr + n;
      int k = 0;
      for (int r = 0; r < h.rows; ++r)
        for (int c = 0; c < h.cols; ++c) {
          const int64_t p = static_cast<int64_t>(r) * h.cols + c;
          if (mask_.host()[p]) { vr[k] = r; vc[k] = c; h_lut_.host()[p] = k++; }
          else h_lut_.host()[p] = -1;
        }
    }
    mark("standardize_and_mask");

    // ---- GPU：上传立方体，融合 kernel 标准化 + 提取 patch，HSI 推理异步发射
    ensure_dev(d_cube_, h.elems());
    ensure_dev(d_stats_, 2 * kBands);
    ensure_dev(d_vrvc_, 2u * n);
    ensure_dev(d_patches_, static_cast<size_t>(n) * kBands * 9);
    ensure_dev(d_y_, static_cast<size_t>(n) * kFeatDim * hsi_.out_elem_bytes());
    ev_hsi0_.record(sa_.get());
    CUDA_CHECK(cudaMemcpyAsync(d_cube_.get(), h_raw_.host(), h.elems() * sizeof(float), cudaMemcpyHostToDevice, sa_.get()));
    CUDA_CHECK(cudaMemcpyAsync(d_stats_.get(), h_stats_.host(), 2 * kBands * sizeof(float), cudaMemcpyHostToDevice, sa_.get()));
    CUDA_CHECK(cudaMemcpyAsync(d_vrvc_.get(), h_vrvc_.host(), 2u * n * sizeof(int32_t), cudaMemcpyHostToDevice, sa_.get()));
    launch_normalize_gather(d_cube_.get(), kBands, h.rows, h.cols, d_stats_.get(), d_stats_.get() + kBands, d_vrvc_.get(),
                            d_vrvc_.get() + n, n, d_patches_.get(), sa_.get());
    hsi_.infer(d_patches_.get(), d_y_.get(), n, sa_.get());
    ev_hsi1_.record(sa_.get());
    if (dbg) {
      dbg->rows = h.rows; dbg->cols = h.cols;
      dbg->mask.assign(mask_.host(), mask_.host() + npix);
      dbg->patches.resize(static_cast<size_t>(n) * kBands * 9);
      CUDA_CHECK(cudaMemcpyAsync(dbg->patches.data(), d_patches_.get(), dbg->patches.size() * sizeof(float), cudaMemcpyDeviceToHost, sa_.get()));
    }
    mark("hsi_patch_and_launch");

    // ---- LAS 段（CPU）。overlap 时这里只是等它结束；否则现在才开始
    if (opt.overlap_las_hsi_read) las_thread.join();
    else las_job();
    if (las_err) std::rethrow_exception(las_err);
    for (auto& kv : las.seg_sec) seg_ms[kv.first] = kv.second * 1000.0;
    t0 = now_ms();  // 上面的 LAS 各段（含重叠的）单独记录，这里重新计时后续步骤

    // ---- PC 推理（第二个 stream）
    const int m = static_cast<int>(las.eligible.size());
    ensure_dev(h_off_, static_cast<size_t>(m) * kNeighbors * 3);
    std::memcpy(h_off_.host(), las.offsets.data(), las.offsets.size() * sizeof(float));
    ensure_dev(d_pc_in_, las.offsets.size());
    ensure_dev(d_pcy_, static_cast<size_t>(m) * kFeatDim * pc_.out_elem_bytes());
    ev_pc0_.record(sb_.get());
    CUDA_CHECK(cudaMemcpyAsync(d_pc_in_.get(), h_off_.host(), las.offsets.size() * sizeof(float), cudaMemcpyHostToDevice, sb_.get()));
    pc_.infer(d_pc_in_.get(), d_pcy_.get(), m, sb_.get());
    ev_pc_.record(sb_.get());

    // ---- 归一化 + 融合窗口匹配（等 HSI、PC 都完成）
    ensure_dev(h_ref_, 2u * m);
    for (int i = 0; i < m; ++i) { h_ref_.host()[i] = static_cast<int32_t>(las.ref_rows[i]); h_ref_.host()[m + i] = static_cast<int32_t>(las.ref_cols[i]); }
    ensure_dev(d_ref_, 2u * m);
    ensure_dev(d_lut_, npix);
    ensure_dev(d_hsi_norm_, static_cast<size_t>(n) * kFeatDim);
    ensure_dev(d_pc_norm_, static_cast<size_t>(m) * kFeatDim);
    ensure_dev(d_out_rows_, static_cast<size_t>(kVariants) * m);
    ensure_dev(d_out_cols_, static_cast<size_t>(kVariants) * m);
    ensure_dev(d_out_cos_, static_cast<size_t>(kVariants) * m);
    ensure_dev(h_out_rows_, static_cast<size_t>(kVariants) * m);
    ensure_dev(h_out_cols_, static_cast<size_t>(kVariants) * m);
    ensure_dev(h_out_cos_, static_cast<size_t>(kVariants) * m);
    CUDA_CHECK(cudaMemcpyAsync(d_ref_.get(), h_ref_.host(), 2u * m * sizeof(int32_t), cudaMemcpyHostToDevice, sa_.get()));
    CUDA_CHECK(cudaMemcpyAsync(d_lut_.get(), h_lut_.host(), npix * sizeof(int32_t), cudaMemcpyHostToDevice, sa_.get()));
    CUDA_CHECK(cudaStreamWaitEvent(sa_.get(), ev_pc_.get(), 0));
    ev_m0_.record(sa_.get());
    launch_normalize_features(d_y_.get(), hsi_out_half(), n, d_hsi_norm_.get(), sa_.get());
    launch_normalize_features(d_pcy_.get(), pc_.out_dtype() == nvinfer1::DataType::kHALF, m, d_pc_norm_.get(), sa_.get());
    launch_match(d_hsi_norm_.get(), d_lut_.get(), d_pc_norm_.get(), d_ref_.get(), d_ref_.get() + m, m, h.rows, h.cols, kSearchRadius,
                 d_out_rows_.get(), d_out_cols_.get(), d_out_cos_.get(), sa_.get());
    CUDA_CHECK(cudaMemcpyAsync(h_out_rows_.host(), d_out_rows_.get(), static_cast<size_t>(kVariants) * m * 4, cudaMemcpyDeviceToHost, sa_.get()));
    CUDA_CHECK(cudaMemcpyAsync(h_out_cols_.host(), d_out_cols_.get(), static_cast<size_t>(kVariants) * m * 4, cudaMemcpyDeviceToHost, sa_.get()));
    CUDA_CHECK(cudaMemcpyAsync(h_out_cos_.host(), d_out_cos_.get(), static_cast<size_t>(kVariants) * m * 4, cudaMemcpyDeviceToHost, sa_.get()));
    if (dbg) {
      dbg->hsi_feat.resize(static_cast<size_t>(n) * kFeatDim);
      dbg->pc_feat.resize(static_cast<size_t>(m) * kFeatDim);
      // 半精度特征展开成 float32：先用归一化前的原始输出，逐位比较更严格
      std::vector<char> yh(static_cast<size_t>(n) * kFeatDim * hsi_.out_elem_bytes()), yp(static_cast<size_t>(m) * kFeatDim * pc_.out_elem_bytes());
      CUDA_CHECK(cudaMemcpyAsync(yh.data(), d_y_.get(), yh.size(), cudaMemcpyDeviceToHost, sa_.get()));
      CUDA_CHECK(cudaMemcpyAsync(yp.data(), d_pcy_.get(), yp.size(), cudaMemcpyDeviceToHost, sa_.get()));
      sa_.sync();
      TrtEngine::expand_to_float(yh.data(), dbg->hsi_feat.size(), hsi_.out_dtype(), dbg->hsi_feat.data());
      TrtEngine::expand_to_float(yp.data(), dbg->pc_feat.size(), pc_.out_dtype(), dbg->pc_feat.data());
      dbg->offsets = las.offsets;
      dbg->ref_rows.assign(h_ref_.host(), h_ref_.host() + m);
      dbg->ref_cols.assign(h_ref_.host() + m, h_ref_.host() + 2 * m);
    }
    ev_m1_.record(sa_.get());
    sa_.sync();
    sb_.sync();
    seg_ms["gpu_tail_pc_match"] = now_ms() - t0;
    // GPU 各段耗时（CUDA event；与上面的 CPU 各段在时间上重叠，不参与求和）
    seg_ms["gpu_hsi_upload_patch_infer"] = Event::elapsed_ms(ev_hsi0_, ev_hsi1_);
    seg_ms["gpu_pc_upload_infer"] = Event::elapsed_ms(ev_pc0_, ev_pc_);
    seg_ms["gpu_normalize_match_download"] = Event::elapsed_ms(ev_m0_, ev_m1_);

    MatchResult r;
    r.n_valid = n;
    r.n_points = m;
    for (int v = 0; v < kVariants; ++v) {
      r.rows[v].assign(h_out_rows_.host() + v * m, h_out_rows_.host() + (v + 1) * m);
      r.cols[v].assign(h_out_cols_.host() + v * m, h_out_cols_.host() + (v + 1) * m);
      r.cosine[v].assign(h_out_cos_.host() + v * m, h_out_cos_.host() + (v + 1) * m);
    }
    return r;
  } catch (...) {
    fail(std::current_exception());
    throw;  // 不会到这里
  }
}
