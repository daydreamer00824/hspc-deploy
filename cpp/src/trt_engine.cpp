#include "trt_engine.h"

#include <cuda_fp16.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "cuda_utils.h"

void TrtEngine::Logger::log(Severity s, const char* msg) noexcept {
  if (s <= Severity::kWARNING) std::fprintf(stderr, "[TRT] %s\n", msg);
}

size_t TrtEngine::dtype_bytes(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kINT32: return 4;
    default: throw std::runtime_error("unsupported tensor dtype");
  }
}

TrtEngine::TrtEngine(const std::string& plan_path, std::vector<int64_t> input_dims, int max_batch,
                     const char* input_name, const char* output_name)
    : in_name_(input_name), out_name_(output_name), in_dims_(std::move(input_dims)), max_batch_(max_batch) {
  std::ifstream f(plan_path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open engine: " + plan_path);
  std::vector<char> blob(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(blob.data(), static_cast<std::streamsize>(blob.size()));

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) throw std::runtime_error("failed to deserialize engine: " + plan_path);
  ctx_.reset(engine_->createExecutionContext());
  if (!ctx_) throw std::runtime_error("failed to create execution context");

  // 张量名不存在或形状不对时，getTensorDataType/getTensorShape 只会返回默认值/垃圾维度而不报错，
  // 后面按这些值分配的缓冲区会与 TRT 实际写入的大小不符。在构造期显式校验，避免留到 infer() 时
  // 才出现未定义行为。
  if (engine_->getTensorIOMode(in_name_.c_str()) != nvinfer1::TensorIOMode::kINPUT)
    throw std::runtime_error("engine has no input tensor named '" + in_name_ + "': " + plan_path);
  if (engine_->getTensorIOMode(out_name_.c_str()) != nvinfer1::TensorIOMode::kOUTPUT)
    throw std::runtime_error("engine has no output tensor named '" + out_name_ + "': " + plan_path);

  in_dtype_ = engine_->getTensorDataType(in_name_.c_str());
  out_dtype_ = engine_->getTensorDataType(out_name_.c_str());
  for (auto d : in_dims_) in_row_ *= d;

  auto id = engine_->getTensorShape(in_name_.c_str());  // d[0] 是动态 batch 维（-1），其余必须与 input_dims 一致
  if (id.nbDims != static_cast<int>(in_dims_.size()) + 1)
    throw std::runtime_error("input tensor rank mismatch: expected " + std::to_string(in_dims_.size() + 1)
                              + " dims (batch + " + std::to_string(in_dims_.size()) + "), got " + std::to_string(id.nbDims) + ": " + plan_path);
  for (size_t i = 0; i < in_dims_.size(); ++i)
    if (id.d[i + 1] != in_dims_[i])
      throw std::runtime_error("input tensor shape mismatch at dim " + std::to_string(i + 1) + ": expected "
                                + std::to_string(in_dims_[i]) + ", engine has " + std::to_string(id.d[i + 1]) + ": " + plan_path);

  auto od = engine_->getTensorShape(out_name_.c_str());
  if (od.nbDims != 2 || od.d[1] <= 0)
    throw std::runtime_error("unexpected output tensor shape (want 2D [batch, dim]): " + plan_path);
  out_dim_ = static_cast<int>(od.d[od.nbDims - 1]);
}

TrtEngine::~TrtEngine() {
  if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
  if (graph_) cudaGraphDestroy(graph_);
}

void TrtEngine::infer(const void* in, void* out, int n, cudaStream_t stream) {
  const auto* ip = static_cast<const char*>(in);
  auto* op = static_cast<char*>(out);
  const size_t in_row_bytes = static_cast<size_t>(in_row_) * in_elem_bytes();
  const size_t out_row_bytes = static_cast<size_t>(out_dim_) * out_elem_bytes();
  for (int off = 0; off < n; off += max_batch_) {
    const int m = std::min(max_batch_, n - off);
    nvinfer1::Dims d;
    d.nbDims = static_cast<int>(in_dims_.size()) + 1;
    d.d[0] = m;
    for (size_t i = 0; i < in_dims_.size(); ++i) d.d[i + 1] = in_dims_[i];
    if (!ctx_->setInputShape(in_name_.c_str(), d)) throw std::runtime_error("setInputShape failed");
    ctx_->setTensorAddress(in_name_.c_str(), const_cast<char*>(ip) + static_cast<size_t>(off) * in_row_bytes);
    ctx_->setTensorAddress(out_name_.c_str(), op + static_cast<size_t>(off) * out_row_bytes);
    if (!ctx_->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed");
  }
}

void TrtEngine::capture_graph(const void* in, void* out, int n, cudaStream_t stream) {
  if (n > max_batch_) throw std::runtime_error("capture_graph: n > max_batch");
  infer(in, out, n, stream);  // 捕获前必须先 eager 跑一次
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  infer(in, out, n, stream);
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
}

void TrtEngine::replay(cudaStream_t stream) { CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream)); }

void TrtEngine::expand_to_float(const void* src, size_t n, nvinfer1::DataType t, float* dst) {
  if (t == nvinfer1::DataType::kFLOAT) std::memcpy(dst, src, n * sizeof(float));
  else if (t == nvinfer1::DataType::kHALF)
    for (size_t i = 0; i < n; ++i) dst[i] = __half2float(static_cast<const __half*>(src)[i]);
  else throw std::runtime_error("expand_to_float: unsupported dtype");
}
