// 最小的 CUDA 工具：错误检查宏与 RAII 封装。
#pragma once
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)                                                                        \
  do {                                                                                          \
    cudaError_t err_ = (expr);                                                                  \
    if (err_ != cudaSuccess)                                                                    \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) + " at " + \
                               __FILE__ + ":" + std::to_string(__LINE__));                      \
  } while (0)

// 设备内存
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) { alloc(n); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { release(); }
  void alloc(size_t n) {
    release();
    n_ = n;
    CUDA_CHECK(cudaMalloc(&p_, n * sizeof(T)));
  }
  void release() {
    if (p_) cudaFree(p_);
    p_ = nullptr;
    n_ = 0;
  }
  T* get() const { return p_; }
  size_t size() const { return n_; }

 private:
  T* p_ = nullptr;
  size_t n_ = 0;
};

// 页锁定 host 内存；mapped=true 时 GPU 可按 dev() 直接访问（Jetson 统一内存下零拷贝）。
template <typename T>
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(size_t n, bool mapped = false) { alloc(n, mapped); }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  ~PinnedBuffer() { release(); }
  void alloc(size_t n, bool mapped = false) {
    release();
    n_ = n;
    CUDA_CHECK(cudaHostAlloc(&h_, n * sizeof(T), mapped ? cudaHostAllocMapped : cudaHostAllocDefault));
    if (mapped) CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_), h_, 0));
  }
  void release() {
    if (h_) cudaFreeHost(h_);
    h_ = nullptr;
    d_ = nullptr;
    n_ = 0;
  }
  T* host() const { return h_; }
  T* dev() const { return d_; }
  size_t size() const { return n_; }

 private:
  T* h_ = nullptr;
  T* d_ = nullptr;
  size_t n_ = 0;
};

class Stream {
 public:
  Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking)); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  ~Stream() { cudaStreamDestroy(s_); }
  cudaStream_t get() const { return s_; }
  void sync() const { CUDA_CHECK(cudaStreamSynchronize(s_)); }

 private:
  cudaStream_t s_{};
};

class Event {
 public:
  Event() { CUDA_CHECK(cudaEventCreate(&e_)); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  ~Event() { cudaEventDestroy(e_); }
  cudaEvent_t get() const { return e_; }
  void record(cudaStream_t s) const { CUDA_CHECK(cudaEventRecord(e_, s)); }
  static float elapsed_ms(const Event& a, const Event& b) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a.e_, b.e_));
    return ms;
  }

 private:
  cudaEvent_t e_{};
};
