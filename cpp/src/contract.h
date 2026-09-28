// 部署链路的数据契约：与 scripts/deploy_scene.py 的 CONTRACT / SPATIAL_CONTRACT 是同一份契约
// （那两个 Python dict 才是唯一可信来源）。这里是它们在 C++ 侧的单一副本，deploy_pipeline.cpp 与
// prep_check_main.cpp 都从这里取值，不再各自重复内嵌字面量。
#pragma once
#include <cstdint>

namespace contract {
constexpr int kBands = 342, kRedIndex = 143, kNirIndex = 262, kPatchSize = 3;
constexpr int kNeighbors = 15, kSearchRadius = 5, kSamplesPerScene = 1000;
constexpr float kNdviThreshold = 0.2f, kNodataEpsilon = 1e-5f;
constexpr uint64_t kSeed = 20260617ULL;
inline constexpr const char* kPointCloudCrs = "EPSG:32651";
inline constexpr const char* kExpectedTargetCrs = "EPSG:4326";
}  // namespace contract
