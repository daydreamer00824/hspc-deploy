// 三个可执行程序（hspc_deploy / hspc_latency / hspc_prep_check）共用的小工具：单调时钟、命令行参数解析、
// 二进制/文本导出、统计量、JSON 布尔字面量。这些和 CUDA/TensorRT 无关，所以单独放在这里，不塞进 cuda_utils.h。
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// 单调时钟。
inline double now_sec() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline double now_ms() { return now_sec() * 1000.0; }

// 极简命令行解析：next() 顺序取出每个参数（可能是 flag 也可能是上一个 flag 的值，由调用方分辨），
// value() 取当前 flag 的值。缺值时报错，而不是像 argv[++i] 那样可能读到 argv 末尾之后。
class ArgParser {
 public:
  ArgParser(int argc, char** argv) : argc_(argc), argv_(argv) {}
  bool next(std::string& flag) {
    if (i_ >= argc_) return false;
    flag = last_ = argv_[i_++];
    return true;
  }
  std::string value() {
    if (i_ >= argc_) throw std::runtime_error("missing value for " + last_);
    return argv_[i_++];
  }

 private:
  int argc_, i_ = 1;
  char** argv_;
  std::string last_;
};

// 各 main 的参数解析都放进 try，缺值（ArgParser）和非法数字（parse_int/parse_double）统一按参数错误以退出码 2 返回。
constexpr int kExitBadArgs = 2;

// 严格数字解析：std::stoi/std::stod 遇到"8x"这类前缀是数字、后面跟着垃圾字符的输入会静默截断、
// 只取前缀数值，不会报错。这里额外检查 pos 覆盖了整个字符串，否则按参数错误处理。
inline int parse_int(const std::string& s) {
  size_t pos = 0;
  int v;
  try {
    v = std::stoi(s, &pos);
  } catch (const std::exception&) {
    throw std::invalid_argument("not an integer: " + s);
  }
  if (pos != s.size()) throw std::invalid_argument("not an integer: " + s);
  return v;
}
inline double parse_double(const std::string& s) {
  size_t pos = 0;
  double v;
  try {
    v = std::stod(s, &pos);
  } catch (const std::exception&) {
    throw std::invalid_argument("not a number: " + s);
  }
  if (pos != s.size()) throw std::invalid_argument("not a number: " + s);
  return v;
}

// 把一个数组按二进制整体写盘（护栏导出用）。打开或写入失败都抛异常，避免缺文件/空文件被
// Python 侧 np.fromfile 读成别的数组、或用旧导出比较，把一次真实的回归误判为"逐位相同"。
template <typename T>
void dump(const std::string& dir, const std::string& name, const std::vector<T>& v) {
  const std::string path = dir + "/" + name;
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open dump file for writing: " + path);
  f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
  if (!f) throw std::runtime_error("failed to write dump file: " + path);
}

// 把一段文本整体写盘，同样的失败即抛异常；meta.txt/shape.txt/wkt.txt/gt.f64 之外，也用于结果 JSON（--out）。
inline void write_text_file(const std::string& path, const std::string& content) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot open file for writing: " + path);
  f << content;
  if (!f) throw std::runtime_error("failed to write file: " + path);
}

// JSON 里的布尔字面量。
inline const char* json_bool(bool v) { return v ? "true" : "false"; }

// 与 numpy.percentile 默认的线性插值一致（q=50 时就是中位数：n 为奇数取中间值，偶数取中间两个的平均）。
inline double percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  const double pos = q / 100.0 * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (v[hi] - v[lo]) * (pos - static_cast<double>(lo));
}
inline double median(const std::vector<double>& v) { return percentile(v, 50); }
