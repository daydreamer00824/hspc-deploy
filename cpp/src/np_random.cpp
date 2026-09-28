#include "np_random.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace {
using u128 = unsigned __int128;

// ---- SeedSequence（numpy/random/bit_generator.pyx）
constexpr uint32_t INIT_A = 0x43b0d7e5, MULT_A = 0x931e8875, INIT_B = 0x8b51f9dd, MULT_B = 0x58f38ded;
constexpr uint32_t MIX_MULT_L = 0xca01f9dd, MIX_MULT_R = 0x4973f715;
constexpr int XSHIFT = 16;

uint32_t hashmix(uint32_t value, uint32_t& hash_const) {
  value ^= hash_const;
  hash_const *= MULT_A;
  value *= hash_const;
  value ^= value >> XSHIFT;
  return value;
}
uint32_t mix(uint32_t x, uint32_t y) {
  uint32_t result = MIX_MULT_L * x - MIX_MULT_R * y;
  result ^= result >> XSHIFT;
  return result;
}

// entropy 为 32 位字数组（整数种子按小端切成 32 位字）；返回 generate_state(4, uint64) 的 4 个 64 位字
void seed_sequence_state(uint64_t seed, uint64_t out[4]) {
  std::vector<uint32_t> entropy;
  if (seed == 0) entropy.push_back(0);
  for (uint64_t s = seed; s > 0; s >>= 32) entropy.push_back(static_cast<uint32_t>(s & 0xFFFFFFFFu));
  uint32_t pool[4];
  uint32_t hc = INIT_A;
  for (size_t i = 0; i < 4; ++i) pool[i] = hashmix(i < entropy.size() ? entropy[i] : 0u, hc);
  for (size_t src = 0; src < 4; ++src)
    for (size_t dst = 0; dst < 4; ++dst)
      if (src != dst) pool[dst] = mix(pool[dst], hashmix(pool[src], hc));
  for (size_t src = 4; src < entropy.size(); ++src)
    for (size_t dst = 0; dst < 4; ++dst) pool[dst] = mix(pool[dst], hashmix(entropy[src], hc));
  uint32_t words[8];
  hc = INIT_B;
  for (int i = 0; i < 8; ++i) {
    uint32_t v = pool[i % 4];
    v ^= hc;
    hc *= MULT_B;
    v *= hc;
    v ^= v >> XSHIFT;
    words[i] = v;
  }
  for (int k = 0; k < 4; ++k) out[k] = static_cast<uint64_t>(words[2 * k]) | (static_cast<uint64_t>(words[2 * k + 1]) << 32);
}

// ---- PCG64（numpy/random/src/pcg64/pcg64.h，XSL-RR 128/64）
class Pcg64 {
 public:
  explicit Pcg64(uint64_t seed) {
    uint64_t s[4];
    seed_sequence_state(seed, s);
    const u128 initstate = (static_cast<u128>(s[0]) << 64) | s[1];
    const u128 initseq = (static_cast<u128>(s[2]) << 64) | s[3];
    state_ = 0;
    inc_ = (initseq << 1) | 1;
    step();
    state_ += initstate;
    step();
  }
  uint64_t next64() {
    step();
    const uint64_t x = static_cast<uint64_t>(state_ >> 64) ^ static_cast<uint64_t>(state_);
    const unsigned rot = static_cast<unsigned>(state_ >> 122);
    return (x >> rot) | (x << ((-rot) & 63));
  }
  uint32_t next32() {  // 先给低 32 位，再给缓存的高 32 位
    if (has_u32_) { has_u32_ = false; return u32_; }
    const uint64_t n = next64();
    has_u32_ = true;
    u32_ = static_cast<uint32_t>(n >> 32);
    return static_cast<uint32_t>(n & 0xFFFFFFFFu);
  }
  // random_bounded_uint64(bitgen, off=0, rng, mask=0, use_masked=False)，只实现 rng <= 2^32-1 的 Lemire 分支
  uint64_t bounded(uint64_t rng) {
    if (rng == 0) return 0;
    if (rng > 0xFFFFFFFFull) throw std::runtime_error("np_random: population larger than 2^32 not supported");
    if (rng == 0xFFFFFFFFull) return next32();
    const uint32_t rng_excl = static_cast<uint32_t>(rng) + 1;
    uint64_t m = static_cast<uint64_t>(next32()) * rng_excl;
    uint32_t leftover = static_cast<uint32_t>(m & 0xFFFFFFFFu);
    if (leftover < rng_excl) {
      const uint32_t threshold = (UINT32_MAX - static_cast<uint32_t>(rng)) % rng_excl;
      while (leftover < threshold) {
        m = static_cast<uint64_t>(next32()) * rng_excl;
        leftover = static_cast<uint32_t>(m & 0xFFFFFFFFu);
      }
    }
    return m >> 32;
  }

 private:
  void step() {
    static const u128 MULT = (static_cast<u128>(2549297995355413924ULL) << 64) | 4865540595714422341ULL;
    state_ = state_ * MULT + inc_;
  }
  u128 state_ = 0, inc_ = 0;
  bool has_u32_ = false;
  uint32_t u32_ = 0;
};
}  // namespace

std::vector<int64_t> np_choice_sorted(uint64_t seed, const std::vector<int64_t>& pop, int64_t size) {
  const int64_t pop_size = static_cast<int64_t>(pop.size());
  if (size > pop_size) throw std::runtime_error("np_choice: sample larger than population");
  Pcg64 rng(seed);
  std::vector<int64_t> idx;
  if (pop_size > 10000 && size > pop_size / 50) {
    // 尾部洗牌：对 arange(pop_size) 做 Fisher-Yates 的后 size 步，取末尾 size 个
    std::vector<int64_t> all(pop_size);
    for (int64_t i = 0; i < pop_size; ++i) all[i] = i;
    for (int64_t i = pop_size - 1; i >= std::max<int64_t>(pop_size - size, 1); --i) {
      const int64_t j = static_cast<int64_t>(rng.bounded(static_cast<uint64_t>(i)));
      std::swap(all[j], all[i]);
    }
    idx.assign(all.begin() + (pop_size - size), all.end());
  } else {
    // Floyd 算法：val 已在集合里就改插 j。之后的 shuffle 不改变集合，调用方随后排序，所以不需要复现
    std::unordered_set<int64_t> seen;
    idx.resize(size);
    for (int64_t j = pop_size - size; j < pop_size; ++j) {
      const int64_t val = static_cast<int64_t>(rng.bounded(static_cast<uint64_t>(j)));
      const int64_t pick = seen.count(val) ? j : val;
      seen.insert(pick);
      idx[j - (pop_size - size)] = pick;
    }
  }
  std::vector<int64_t> out(size);
  for (int64_t i = 0; i < size; ++i) out[i] = pop[idx[i]];
  std::sort(out.begin(), out.end());
  return out;
}
