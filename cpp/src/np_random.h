// NumPy 随机数的逐位复现：default_rng(seed).choice(pop, size, replace=False)，只需要"选中的集合"（调用方随后排序）。
// SeedSequence → PCG64（XSL-RR 128/64）→ Lemire 有界整数 → Floyd 算法 / 尾部洗牌（与 NumPy 的选择规则一致）。
#pragma once
#include <cstdint>
#include <vector>

// 返回选中的元素（升序）。pop 需为升序整数数组以外的任意数组时，仍按"取值后排序"处理。
std::vector<int64_t> np_choice_sorted(uint64_t seed, const std::vector<int64_t>& pop, int64_t size);
