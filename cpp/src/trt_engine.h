// TensorRT C++ 运行时封装：反序列化 .plan、按 max_batch 分块 enqueueV3、可选 CUDA Graph。
#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class TrtEngine {
 public:
  // input_dims 为单个样本的形状（不含 batch），如 HSI {342,3,3}、PC {15,3}。
  TrtEngine(const std::string& plan_path, std::vector<int64_t> input_dims, int max_batch,
            const char* input_name = "input", const char* output_name = "output");
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  nvinfer1::DataType in_dtype() const { return in_dtype_; }
  nvinfer1::DataType out_dtype() const { return out_dtype_; }
  size_t in_elem_bytes() const { return dtype_bytes(in_dtype_); }
  size_t out_elem_bytes() const { return dtype_bytes(out_dtype_); }
  int64_t in_row_elems() const { return in_row_; }
  int out_dim() const { return out_dim_; }
  int max_batch() const { return max_batch_; }
  static size_t dtype_bytes(nvinfer1::DataType t);

  // 把一段 float32/float16 的 TensorRT 输出缓冲区逐元素转换成 float32，写进 dst（必须能容纳 n 个 float）。
  // float32 源直接 memcpy；float16 按 IEEE 转换。deploy_pipeline.cpp 和 latency_main.cpp 共用。
  static void expand_to_float(const void* src, size_t n, nvinfer1::DataType t, float* dst);

  // 在设备可访问地址上整批推理（cudaMalloc / 映射 host 内存都行）。n > max_batch 时按 max_batch 分块，
  // 只是移动地址偏移，没有任何拷贝。只下发不同步，同步由调用方负责。
  void infer(const void* in, void* out, int n, cudaStream_t stream);

  // 固定 batch 的 CUDA Graph：先 eager 跑一次（TRT 惰性初始化），再捕获 enqueueV3；之后 replay() 复用。
  void capture_graph(const void* in, void* out, int n, cudaStream_t stream);
  void replay(cudaStream_t stream);
  bool has_graph() const { return graph_exec_ != nullptr; }

 private:
  struct Logger : nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override;
  } logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
  std::string in_name_, out_name_;
  std::vector<int64_t> in_dims_;
  int64_t in_row_ = 1;
  int out_dim_ = 0;
  int max_batch_ = 0;
  nvinfer1::DataType in_dtype_{}, out_dtype_{};
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
};
