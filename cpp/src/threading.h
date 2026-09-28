// 极简并行 for：把 [0, n) 切成连续的块，交给 workers 个线程。每个下标的结果互不依赖，所以线程数不影响结果。
#pragma once
#include <algorithm>
#include <cstdint>
#include <exception>
#include <thread>
#include <vector>

// fn(worker_index, begin, end)：worker_index 是这个线程分到的第几块（从 0 开始，单线程回退时恒为 0），
// 供需要"每个线程一份独立资源"的调用方按下标索引（如 las_pipeline.cpp 给每个投影线程分配的 PJ 上下文）。
// 工作线程里抛出的异常会被捕获，所有线程 join 完之后在调用方所在线程重新抛出第一个，而不是让异常
// 逃出线程函数直接 std::terminate。
template <typename F>
void parallel_chunks(int64_t n, int workers, F&& fn) {
  if (workers <= 0) workers = static_cast<int>(std::thread::hardware_concurrency());
  workers = static_cast<int>(std::min<int64_t>(workers, std::max<int64_t>(n, 1)));
  if (workers <= 1) { fn(0, int64_t{0}, n); return; }
  std::vector<std::thread> th;
  std::vector<std::exception_ptr> errs(workers);
  const int64_t chunk = (n + workers - 1) / workers;
  for (int w = 0; w < workers; ++w) {
    const int64_t b = w * chunk, e = std::min(n, b + chunk);
    if (b >= e) break;
    th.emplace_back([&fn, &errs, w, b, e] {
      try { fn(w, b, e); } catch (...) { errs[w] = std::current_exception(); }
    });
  }
  for (auto& t : th) t.join();
  for (auto& e : errs) if (e) std::rethrow_exception(e);
}

template <typename F>
void parallel_for(int64_t n, int workers, F&& fn) {  // fn(begin, end)
  parallel_chunks(n, workers, [&fn](int, int64_t b, int64_t e) { fn(b, e); });
}
