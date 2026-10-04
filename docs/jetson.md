# Jetson Orin Nano Super 端侧部署报告

本文记录把 x86 上的 PyTorch → ONNX → TensorRT 流程移植到 Jetson Orin Nano Super，在板上做整景推理加速，并给出从原始文件到匹配结果的完整端到端数字。
**文中所有端侧数字都在开发板上实测**；x86（RTX 3060）数字只在标明的地方作为不同平台的对照出现。原始结果都在 `results/jetson/`。

## 环境与工作方式

| 项 | 值 |
|---|---|
| 硬件 | Jetson Orin Nano Engineering Reference Developer Kit Super；GPU SM 8.7、8 个 SM；6 核 ARM CPU；CPU/GPU 共享 7.4 GiB 内存 |
| 软件 | L4T R36.4.7（JetPack 6）、TensorRT 10.7.0、CUDA 12.6、PyTorch 2.5.0（NVIDIA Jetson 版）、Python 3.10 |
| 功耗模式 | MAXN_SUPER；测试期间使用 `jetson_clocks` 锁定时钟（CPU 1.728GHz、GPU 1.02GHz、EMC 3.2GHz）；所有数字都在这个状态下测得 |

- 代码在本机写、用 git 管理，用 rsync 单向推到板上运行，结果拉回 `results/jetson/`。板上只有运行副本，不改动板上任何已有环境。
- `.plan` 与 GPU 架构、TRT 版本绑定，x86 的引擎不能用；板上用同一份 ONNX 重建。交付引擎的构建配置与 sha256 见 `results/jetson/engine_manifest.json`。
- 移植原则：先走 TensorRT 的常规默认路线，x86 上为特定问题加过的补救手段（选择性混合精度、大 batch profile、显式 QDQ）只在板上复现了同样的问题时才引入。

**两个 Python 环境**：

- 推理与优化实验（第 1～3 节）：系统 Python 3.10（自带 TensorRT 绑定和 Jetson 版 torch），另外只用 `pip --target` 在项目目录里装了 polygraphy。
- 完整端到端（第 4 节，需要 GDAL/pyproj/laspy）：独立 conda 环境 `hspc-jetson`，版本与 x86 的前处理环境对齐（gdal 3.11.4、pyproj 3.7.1 / PROJ 9.6.2、laspy 2.7.0、scipy 1.15.2），
  只有 numpy 用 1.26（Jetson 版 torch 不兼容 numpy 2）；再装板上现成的 Jetson torch wheel 与 polygraphy，TensorRT 绑定从系统 `dist-packages` 复制进环境。
  运行时要设 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`（否则先 import tensorrt 会带入系统旧版 libstdc++，scipy 随后报 `GLIBCXX_3.4.32 not found`）、
  `PYTHONNOUSERSITE=1`（`~/.local` 里的包对任何 python3.10 都可见）、`PROJ_DATA/GDAL_DATA` 指向环境的 share 目录。包清单见 `results/jetson/logs/env_hspc-jetson.txt`。
  两个环境跑同一个配置，特征与匹配结果逐位相同（`logs/p1_env_check.log`）。

## 1. 精度与单样本性能

**FP32 基准**：板上 PyTorch FP32 与 x86 CPU FP32 参考的差异 PC ≤1.1e-5、HSI ≤6.6e-5（`j1_cross_platform_fp32.json`）。引擎精度验证以 x86 CPU FP32 为参考，与 x86 阶段保持同一精度口径；Jetson 整景任务级匹配一致率则对照板上 PyTorch FP32。

**引擎精度**（后 200 条验证集，与校准集不重叠；`stage2_trt_accuracy.json`、`j2_*.json`）：

| 模型 / 模式 | cos_sim_min | 同模态最近邻 Top-1 | 说明 |
|---|---|---|---|
| PC / HSI FP32（TF32 off） | 1.000000 / 1.000000 | 100% / 100% | 闸门通过：`cos_sim_min ≥ 0.9999` 且 `max_abs_err ≤ 1.5e-4`；实测 max_abs 为 1.0e-5 / 1.0e-4 |
| PC FP16（默认精度） | 0.999993 | 100% | 3 次独立构建输出逐位相同，未触发补救 |
| HSI FP16（默认精度） | 0.990～0.996 | 97.5%～98.0% | 3 次构建之间 max_abs 相差 0.114：复现了 x86 上的构建不稳定 |
| **HSI FP16 mixed（stem 组保留 FP32）** | **0.999705** | **99.0%** | 触发补救后采用；3 次构建指标相同 |

**单样本延迟**（`stage3_benchmark.json`，p50，warmup 50 / 测 300，CUDA event 计时；`trtexec` 交叉核对偏差 <3%）：

| 模型 | PyTorch FP32 b1 | TRT FP32 b1 | **TRT FP16 b1** | TRT FP16 b64 | b64 吞吐 | 加速比（b1） |
|---|---|---|---|---|---|---|
| PC | 4.652 ms | 0.761 ms | **0.328 ms** | 1.173 ms | 54,552 条/s | 14.2x |
| HSI | 5.271 ms | 0.718 ms | **0.355 ms** | 0.968 ms | 66,097 条/s | 14.8x |

**CUDA Graph**（`opt_o4_latency.json`）：`trtexec --useCudaGraph` 显示 GPU 段快 30% 以上，于是在 Python 里实现固定形状、捕获一次反复 replay 的推理（`scripts/jetson_scene_opt.py` 的 `GraphRunner`），输出与不用 Graph **逐位相同**：

| 配置 | 仅 GPU 段（event） | 含 pinned H2D/D2H 的完整请求 |
|---|---|---|
| PC b1 | 0.328 → 0.230 ms（-30%） | 0.492 → 0.410 ms（-17%） |
| HSI b1 | 0.355 → 0.257 ms（-28%） | 0.500 → 0.440 ms（-12%） |
| PC b8 / HSI b8 | -27% / -27% | -14% / -11% |

单样本时 GPU 计算只有约 0.3ms，而 ARM CPU 每次要下发约 36 个 kernel、每个 7～10µs，Graph 消掉的就是这部分下发间隙；把同步和拷贝都算进去，真实请求快 11%～17%。

## 2. INT8：板上没有可交付的 INT8

- **隐式校准路径没有真正执行 Int8**：板上使用 `trtexec --int8 --calib=<缓存> --exportLayerInfo` 逐层核查，PC / HSI 的隐式校准引擎均为 0 个 Int8 层（`results/jetson/int8_implicit_layer_check.json`）。因此该路径不作为 INT8 性能结果报告。
- **x86 复核结论一致**：使用同一构建脚本对 PC / HSI 各独立重建 3 次，所有构建仍为 0 个 Int8 层（`results/x86_int8_implicit_verification.json`）。不同构建之间仍可能因 TensorRT tactic 选择表现出 FP16 / auto 路径上的数值差异，但没有构建真正进入 Int8。
- **显式 QDQ 才真正产生 Int8 层**：Jetson 上 HSI 的 3 次构建均有 30 个 Int8 层，PC 均有 32 个 Int8 层（`results/jetson/j2_qdq_stability.json`）。
- **HSI QDQ**：TensorRT 中 cos_sim_min 约 0.755；同一个 QDQ ONNX 模型直接使用 ONNX Runtime 执行时也出现明显精度下降，因此 HSI 的主要精度问题在 TensorRT 执行之前已经存在于量化后的 QDQ 模型中。
- **PC QDQ**：Jetson 上 3 次独立 TensorRT 构建均稳定复现严重数值异常（cos_sim_mean 约 -0.089，max_abs 约 11.886）。当前公开 evidence 中没有 PC QDQ 的 ONNX Runtime direct 对照，因此现有证据不足以进一步区分问题来自 QDQ 图本身还是 TensorRT 对该图的执行路径。
- HSI QDQ 在当前 batch=64 测试中延迟由 0.968ms 降至 0.915ms，降低约 5.5%（对应约 1.058x）。综合精度和性能结果，本项目没有形成可交付 INT8，正式部署继续采用 FP16 / selective mixed precision。

## 3. 整景推理链路加速（缓存输入）

### 起点与瓶颈

起点 S0 是 x86 阶段 4 的“验证工具”写法：整个网格 15,840 个 patch 全部推理、torch 版 runner（每次调用重新分配显存）、batch64 引擎、torch 逐点循环匹配。
nsys 时间线（独立 profiling run，计时口径见下文“GPU 时间线”；`logs/nsys/o0_S0_*`）显示一次整景约 2.7s 里 **GPU 只忙了 9.6%**：61.6% 在 CPU 逐点匹配，16% 在 CPU 构建 patch；HSI 推理窗口里 GPU 也有 55% 在空等每次调用的 CPU 开销。

### 累加式单变量对比

同一进程内，配置逐个累加、每步只改一处；每个配置 warmup 2 + 重复 10 次取中位数，正反各跑一轮。场景 24data/10.6/1（9,645 个有效 HSI 像元、1,000 个点）。
本节计时从缓存的原始 HSI 立方体开始（标准化、掩膜、patch、推理、拼网格、匹配），文件读取与 LAS 段见第 4 节。结果 `opt_final2_compare.json`，下表就是 README 累加图的数据：

| 配置 | 改动 | prep | patch_build | hsi_infer | pc_infer | assemble | match | 总计 ms（正/反） | 相对 S0（正/反） |
|---|---|---|---|---|---|---|---|---|---|
| S0 | 验证工具式写法 | 24.5 | 428.7 | 428.3 | 29.6 | 0.0 | 1364.3 | 2280.0 / 2299.9 | 1.00x |
| S1 | 只推理有效像元 | 24.5 | 234.0 | 285.5 | 35.2 | 11.1 | 1279.4 | 1855.9 / 1911.9 | 1.23x / 1.20x |
| S2 | NumPy 合并匹配（x86 阶段 7 的实现） | 24.4 | 236.7 | 241.8 | 29.3 | 11.0 | 250.3 | 795.4 / 785.2 | 2.87x / 2.93x |
| S3 | 不依赖 torch、buffer 复用的 runner | 24.4 | 237.9 | 255.5 | 29.3 | 11.3 | 249.7 | 809.9 / 800.4 | 2.82x / 2.87x |
| S4 | 大 batch 引擎（profile 1024） | 24.4 | 236.1 | 162.3 | 17.9 | 11.1 | 249.4 | 702.8 / 694.4 | 3.24x / 3.31x |
| S5 | 统一内存零拷贝 | 24.4 | 193.7 | 106.5 | 16.8 | 11.3 | 245.4 | 600.3 / 598.1 | 3.80x / 3.85x |
| S6 | 匹配搬到 GPU | 24.5 | 193.5 | 106.8 | 16.7 | 11.3 | 57.4 | 411.5 / 411.4 | 5.54x / 5.59x |
| S7 | HSI patch 在 GPU 上 gather | 24.4 | 13.9 | 106.5 | 16.7 | 11.1 | 57.3 | 230.8 / 231.8 | 9.88x / 9.92x |
| S8 | 特征留在 GPU（拼网格、匹配不回 host） | 24.2 | 13.9 | 107.3 | 16.1 | 0.4 | 49.8 | 212.2 / 212.3 | 10.75x / 10.83x |
| S9 | PC 推理放第二个 stream，与 CPU 标准化重叠 | 28.2 | 13.9 | 106.8 | 0.9 | 0.4 | 49.7 | 200.2 / 200.8 | 11.39x / 11.45x |
| S10 | 匹配不分块（1,000 点一次 gather + bmm） | 28.3 | 13.9 | 106.7 | 0.8 | 0.4 | 39.7 | 190.4 / 191.0 | 11.97x / 12.04x |
| **S11b** | 逐元素标准化放到 GPU（归约仍用 NumPy） | 23.9 | 9.7 | 107.3 | 0.7 | 0.3 | 39.5 | **181.8 / 182.0** | **12.54x / 12.63x** |

- 分段是各自的中位数，相加不严格等于总计中位数；S9 起 pc_infer 只是发射时间，推理与 prep 重叠。
- S0 本身在不同批次间有波动（2190～2280ms：CPU 单线程逐点循环），对外只引用同一批次内的加速比，并取正向的 12.54x。
- S3 没有变快（与 x86 阶段 7 一致），它的价值是部署链路不依赖 torch；保留在链条里展示。

### 各项优化做了什么

S1 和 S2 沿用 x86 阶段已有的整景链路优化，其中 S1 → S2 从 1855.9ms 降至 795.4ms，是整个累加链中最大的单步下降。S3 之后主要考察 Jetson 上的 Runtime、统一内存和 GPU 数据通路优化。

- **统一内存零拷贝（S5）**：Jetson 的 CPU 和 GPU 是同一块 DRAM。输入输出改用 `cudaHostAlloc(cudaHostAllocMapped)` 分配、`cudaHostGetDevicePointer`
  取得 GPU 地址，patch 直接写进这块内存，TensorRT 按偏移直接读写（`deploy_scene.py`：`PinnedArray(mapped=True)`、`LiteTrtRunner.infer_ptrs`）。
  收益约 100ms，远大于 nsys 里 memcpy 本身的 12ms：同时省掉了 `np.stack` 之后再整体拷贝的一次 118MB host 内存搬运。
  x86（独立显卡 + PCIe）上 pinned 内存没有稳定收益（README 技术要点第 2 条）；Jetson 统一内存下，映射内存还同时省掉 host 端整理拷贝，因此收益明显。
- **GPU 匹配（S6）**：搜索窗口四周补边后固定为 11×11，1,000 个点按块 gather 成 (点数, 121, 1024) 再做一次 bmm，行优先取第一个最大值，
  与逐点循环的 NumPy 版行列、cosine、像元位移全部逐位相同（`deploy_jetson.py` 的 `gpu_score_window_multi`）。
- **GPU 上提取 patch（S7，关键 GPU 数据通路优化）**：标准化后的立方体（21.7MB）上传一次，四周补一圈零（与 `extract_patch` 的零填充一致），
  用预先算好的 3×3 邻域索引一次 gather 出 (N, 342, 3, 3)，直接作为 TensorRT 输入地址。原来这里是 CPU 单线程 Python 循环（tegrastats 显示 6 核只有 1 核跑满）。
- **特征留在 GPU（S8）**：TensorRT 输出写进 torch 设备张量，在 GPU 上 scatter 成网格并直接匹配，最后只把 1,000 个点的结果取回 host。
- **双 stream（S9）**：PC 推理不依赖 HSI，在 CPU 做标准化之前先发射到第二个 stream。
- **匹配不分块（S10）**：固定真实特征的微基准里，分块 128/256/512/1000 → 58.6/46.5/40.1/36.9ms，全部逐位相同（`opt_matchbench.json`）；匹配段峰值显存约 600～700MB。
- **逐元素标准化上 GPU（S11b）**：均值、标准差、NDVI 掩膜的归约仍用 NumPy，只把 (raw − mean) / std 放到 GPU，直接接 GPU gather——
  IEEE 逐元素减法与除法结果确定，所以 4 景都与 S10 逐位相同，同时省掉了标准化结果的上传。
- **显存要及时释放**：HSI 推理完立刻释放 patch（119MB）与补边立方体。一次装完环境后空闲内存只剩 1.4～1.7GB（其余是页缓存）时，
  不释放会与匹配段的窗口 gather 叠加，触发 `NvMapMemAllocInternalTagged ... error 12`（ENOMEM）；释放后同样条件下正常，计算结果不变。

### 没有采纳的候选（4 景、采纳条件事先定好，`opt_p25_*.json`）

| 候选 | 默认景 ms | 结果 | 判定 |
|---|---|---|---|
| S11：标准化与掩膜整体放 GPU（归约也在 GPU） | 190 → 173 | 掩膜逐位相同，但 4 景里 3 景匹配行列变了（近平局翻转，特征差 0.03～0.05） | 不采纳：条件是 4 景行列不变 |
| S10T：HSI 引擎 stem 组允许 TF32 | 190 → 190 | hsi_infer 只快 0.3%；3 次构建中 1 次与另两次差 0.039 | 不采纳：提速 <3%，且 3 次构建结果不一致 |

S11 说明了一点：输入末位级的差异经过 FP16 引擎会放大成 0.03～0.05 的特征差，足以翻转近平局；S11b 把归约留在 CPU，拿到了大部分收益并保持逐位相同。

### 护栏

- 前处理：板上重建的全网格 patch、有效像元掩膜与缓存逐位相同；每个配置算出的掩膜也与缓存逐位相同。
- 相邻配置：除 S4 换引擎外，每一步的 HSI/PC 特征和两个匹配变体的行列都与上一步**逐位相同**；S4 特征最大差 0.036，匹配行列仍相同。
- 对板上 PyTorch FP32：每个配置都是 DTP 5 个、DTP+Spatial 3 个不一致（一致率 99.5% / 99.7%），**全部是近平局翻转**（FP32 top1-top2 分差最大 0.00065，判据 0.0096）。

### GPU 时间线：9.6% → 91.4%

nsys 的独立 profiling run 中，S0 一次整景约 2.7s，GPU 忙碌约 258.8ms（9.6%）；S11b 一次整景 188.7ms，GPU 忙碌 172.5ms（91.4%）。S11b 的 HSI 推理窗口 GPU 忙碌率约 99%。

这里的 nsys 数据用于分析 GPU 活跃区间和时间线，不作为正式延迟统计；第 3 节累加表中的 2280.0ms → 181.8ms 来自不启用 profiler 的同进程重复测量中位数。因此两组绝对耗时不要求完全一致，现有 evidence 也不把全部差异简单归因于 profiler 开销。

剩下的约 107ms HSI 推理是当前 TensorRT 版本、引擎、输入形状和 MAXN_SUPER 功耗模式下观察到的主要瓶颈：`builderOptimizationLevel=5` 重建后 GPU 时间基本不变（108.0 vs 107.7ms），构建耗时却从 47s 增至 257s，因此该方案未采纳（`opt_optlevel.json`）。

### 多场景验证

另取 3 个未参与默认景优化调参、覆盖两个年份的场景。每景同进程 S0 与 S11b 正反各一轮（`opt_final2_multiscene_*.json`）：

| 场景 | 有效像元 | S0 ms（正/反） | S11b ms（正/反） | 加速比（正/反） | 对 FP32 不一致数 DTP / DTP+Spatial | 最大分差 | HSI 特征 cos_sim_min |
|---|---|---|---|---|---|---|---|
| 24data/10.6/1 | 9,645 | 2177 / 2171 | 180.9 / 181.5 | 12.04x / 11.96x | 5 / 3 | 0.00065 | 0.999072 |
| 24data/10.6/10 | 7,404 | 2343 / 2516 | 159.5 / 159.4 | 14.69x / 15.79x | 3 / 6 | 0.00089 | 0.999671 |
| 25data/8.4/1 | 6,808 | 2213 / 2213 | 144.1 / 144.8 | 15.36x / 15.28x | 8 / 2 | 0.00068 | 0.998787 |
| 25data/10.22/1 | 8,077 | 2072 / 2258 | 161.4 / 162.0 | 12.84x / 13.94x | 3 / 4 | 0.00037 | 0.999774 |

4 景加速 12.0～15.4x（正向），前处理与掩膜护栏全部通过，不一致点全部是近平局。这里默认景的 12.04x 来自后续 4 景独立验证批次；前文 12.54x 来自默认景同进程累加实验，两者不是同一次测量。25data/8.4/1 的 HSI 特征 cos_sim_min 为 0.998787，
S0 与 S11b 相同（6,808 个 patch 上取最小值，比 200 条验证集更容易碰到尾部），不是优化造成的，任务级仍全部近平局。

## 4. 从原始文件到匹配结果（完整端到端）

部署入口 `scripts/deploy_jetson.py`：读原始 HSI（GDAL）→ 标准化 + 掩膜 → GPU gather → TensorRT HSI（异步）→ LAS 读取、投影、kNN（CPU，与 GPU 推理重叠，
kd 树构建放后台线程与投影并行，kNN 查询用全部核）→ TensorRT PC（第二个 stream）→ GPU 拼网格 + 匹配。默认参数即最终配置。

**前处理护栏**：4 景在板上从原始文件重新生成的缓存，与 x86 生成的逐数组比较（patch、原始立方体、掩膜、point_offsets、ref_rows/cols、truth、采样索引等 11 个数组），
**全部逐位相同**（`preprocess_guardrail.json`）——ARM 上的 GDAL/PROJ/laspy/cKDTree 与 numpy 1.26（x86 为 2.2.6）不影响结果。

**对比**：每景 5 种配置，各自独立进程，正反两轮（汇总 `e2e_full_summary.json`、`e2e_las_summary.json`；每次运行的原始 JSON 含逐点匹配输出，只保留在本地）。**每一次运行的匹配行列都与缓存输入的 S11b 逐位相同。**

| 场景 | A：x86 最终链路原样（`deploy_scene.py`） | B：Jetson 链路顺序执行 | C：+ LAS 与 GPU 推理重叠 | D：+ kNN 查询多核 | **E：+ kd 树构建与投影并行** | A→E（保守下界） |
|---|---|---|---|---|---|---|
| 24data/10.6/1 | 921.5 / 920.9 | 376.9 / 374.9 | 290.2 / 290.9 | 285.4 / 285.4 | **256.7 / 254.8** | ≥3.58x |
| 24data/10.6/10 | 861.2 / 859.2 | 396.0 / 396.4 | 325.6 / 327.8 | 322.2 / 320.6 | **266.1 / 267.8** | ≥3.23x |
| 25data/8.4/1 | 741.2 / 734.5 | 299.9 / 303.2 | 239.1 / 242.5 | 238.7 / 238.6 | **213.5 / 216.9** | ≥3.47x |
| 25data/10.22/1 | 801.7 / 805.7 | 323.3 / 322.9 | 250.2 / 249.1 | 247.2 / 247.3 | **229.2 / 231.6** | ≥3.49x |

（单位 ms，正/反两轮。**聚合方式**：A～E 都是 10 次运行取中位数。**计时范围**：A（`deploy_scene.py`）顺序执行，
每次整景耗时取该次各分段之和；B～E 是整景外层墙钟（`pipe.run` 前后同步计时）。**为什么是下界**：A 的分段计时
都在主线程上顺序发生、互不重叠，不含分段之间未计时的代码，所以每次运行的分段之和 ≤ 该次真实整景墙钟；中位数
对逐项不等式单调，所以 A 的 `total_ms` ≤ A 真实整景墙钟的中位数；以 E 的真实墙钟为分母时，A÷E ≤ 真实加速比，
即表中数字是保守下界。当时没有记录 A 的外层墙钟，不重测无法恢复精确值。倍数向下截断到两位小数，保证展示值
不高于原始比值；默认景原始比值约 3.589994x，因此显示 ≥3.58x，而不是 3.59x。`deploy_scene.py` 自身报告的
"各段中位数之和"（先取每段中位数再相加，聚合顺序与上面相反）保留在 `e2e_full_summary.json` 的
`total_ms_sum_of_segment_medians`，只用于来源追溯：它与当前 A 值最多相差 1.3ms，但这个差值是两种不同聚合顺序
的差，不能当作"分段之间未计时代码耗时"的估计。A～C 与 D～E 分两批跑，第二批里复测的 C（`e2e_las_summary.json`
记为 C2）与第一批几乎相同，如 290.2 vs 290.3ms，批次间没有漂移。）

- **A → B**：A 的大头是 CPU 构建 patch（252ms）、NumPy 匹配（268ms）和 pageable 拷贝下的 HSI 推理（164ms），连同 PC 推理与拼网格共约 713ms；
  B 用第 3 节的 GPU 链路把这部分压到约 173ms（HSI patch 与推理 116ms + PC 推理、拼网格与匹配 57ms）。
- **B → C**：HSI 推理异步发射后直接做 LAS 段，GPU 推理几乎完全藏在 CPU 工作后面（等 GPU 只剩 1.4ms）。此时 **LAS 段（含读 LAS）148ms、占 51%，成为瓶颈**（投影 59、kd 树构建 50、查询与组装 25ms，都是单线程）。
  重叠不是免费的：同样的投影在 B 里 49ms、在 C 里 59ms，kd 树构建 42 → 50ms——CPU 与 GPU 同时运行时在争抢同一块共享内存的带宽，这是统一内存架构上 CPU/GPU 并行的代价。
- **C → D**：kNN 查询改用全部核，只快 1～5ms——这一段的时间主要是逐点组装邻域偏移的 Python 循环，不是查询本身。
- **D → E**：kd 树构建（只依赖原始坐标）放到后台线程，与投影并行，完全藏在投影后面（等待 0.1～0.2ms），快 10%～18%。
  默认景的关键路径随之转到 GPU：等 HSI 推理的时间从 1.4ms 变成 28.7ms，再缩短 LAS 段已经没有收益。

E 的默认景分段（ms）：读 HSI 57.9、标准化与掩膜 19.5、HSI patch 与发射 8.4、读 LAS 7.4、CRS 初始化 1.4、投影 61.4、掩膜过滤与采样 4.8、
等 kd 树 0.2、kNN 查询与组装 26.0、等 HSI 推理 28.7、PC 推理与拼网格匹配 40.3。

## 5. C++ 部署（不依赖 Python / torch）

前面各节的部署入口都是 Python（`deploy_jetson.py`：TensorRT Python 绑定 + torch 做 GPU 算子）。实际端侧部署一般是一个 C++ 可执行文件，不带 Python 和 libtorch，
所以 `cpp/` 下用 C++ 把整条链路重做了一遍：读文件、前处理、推理、匹配全部在一个进程里完成。Python 版保留，作为**参考实现**（逐位对照）和**基线**（同一块板、同一场次对比）。
加载的是同一批 `.plan` 引擎，因此两个编码器本身使用相同的 TensorRT engine；Python 与 C++ 版本的整景差异则同时来自 Runtime 调度、CPU 前处理、I/O 和自定义 CUDA kernel 等实现差异。

| 模块 | 内容 |
|---|---|
| `cpp/src/trt_engine.*` | TensorRT C++ 运行时：反序列化 `.plan`、按 max_batch 分块 `enqueueV3`、可选 CUDA Graph（`cudaStreamBeginCapture` 包住 `enqueueV3`，捕获一次、反复 replay） |
| `cpp/src/kernels.cu` | 自定义 CUDA kernel：**标准化 + patch 提取融合**（直接从原始立方体算 `(raw-mean)/std` 并写成 `(N,342,3,3)`，不生成整幅标准化立方体）、特征归一化、**融合窗口匹配**（每个参考点一个 block，按有效像元索引表直接读 HSI 特征，不拼 `rows×cols×1024` 的整幅网格） |
| `cpp/src/hsi_io.*`、`hsi_prep.*` | GDAL C API 读 HSI（含整块读快速路径）、NDVI 掩膜、按波段均值/标准差 |
| `cpp/src/las_pipeline.*` | LAS 读取、PROJ C API 投影（可多线程）、采样、k 近邻与邻域偏移 |
| `cpp/src/np_random.*` | NumPy `default_rng(seed).choice(..., replace=False)` 的逐位复现（SeedSequence → PCG64 → Lemire 有界整数 → Floyd / 尾部洗牌） |
| `cpp/third_party/scipy_ckdtree/` | SciPy 1.15.2 的 cKDTree 建树与 kNN 查询内核（BSD-3，与原文件逐字节相同，只改了 `ckdtree_decl.h` 里对 numpy 头文件的依赖） |
| `cpp/src/deploy_pipeline.*`、`deploy_main.cpp` | 整景链路与命令行程序 `hspc_deploy`；`latency_main.cpp`：单样本延迟 `hspc_latency`；`prep_check_main.cpp`：CPU 前处理护栏用 |

匹配打分规则是课题方法本身，放在不入库的 `cpp/src/matching_rule.cuh`（与 `scripts/matching.py` 是同一条规则）。
公开版本缺少该文件时，CMake 会跳过 `hspc_deploy`，不依赖它的 `hspc_latency` 和 `hspc_prep_check` 仍可构建。

**构建**（板上原生编译；GDAL/PROJ 用 `hspc-jetson` conda 环境里的同一份库，运行时用其 libstdc++）：

```bash
cd cpp && cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DHSPC_CONDA_PREFIX=$CONDA_PREFIX && cmake --build build -j4
```

编译选项里有两类浮点控制对逐位一致很关键：CPU 端使用 `-ffp-contract=off`，避免 aarch64 GCC 将 `a*b+c` 自动融合成 FMA；CUDA 端不开 `--use_fast_math`，保持精确除法等默认浮点行为，同时显式使用 `-fmad=false` 禁止乘加融合。这样可以与 NumPy / laspy 的两步舍入口径保持一致。

### 逐位复现：怎么做到、怎么验证

护栏在 `scripts/cpp_guardrail.py`：C++ 程序把中间数组按二进制导出，Python 参考链路（`preprocess.py`、`deploy_scene.py`、`deploy_jetson.py`）逐数组比较。**4 景全部通过：**

| 阶段 | 逐位相同的数组（4 景） |
|---|---|
| CPU 前处理（`prep-check`） | geotransform 与 WKT、原始立方体、有效像元掩膜、342 个波段的均值与标准差、点云 xyz、PROJ 投影后的坐标（float64）、全部点的像元行列、采样后的点下标、参考行列、邻域偏移 `point_offsets` |
| 整景链路（`deploy-check`） | patch、HSI 特征、PC 特征、两个 variant 的匹配行列 |
| 每一次性能测量运行 | 匹配行列与同一景的 Python 版（P）逐位相同；P + K0～K5、正反两轮、4 景共 56 个配置轮次，其中 C++ K0～K5 为 48 个配置轮次（`cpp_e2e_summary.json`），每个进程内的 10 次重复也相同 |

其中需要说明的几处：

- **均值与标准差**：NumPy 对 `raw.mean(axis=(1,2))` 的实际求和顺序是"每 8192 个元素一块（NumPy 默认缓冲区大小），块内做 pairwise，块与块之间顺序累加"，不是对整幅做一次 pairwise；
  按后者写的第一版有约三成波段的均值与 NumPy 差最后一位，改成分块口径后 4 景全部逐位相同。这个口径对应板上的 numpy 1.26.4。
- **投影**：链接的是与 pyproj 同一份 libproj（PROJ 9.6.2），按 pyproj 的 `always_xy=True` 同样调用 `proj_normalize_for_visualization`，投影后的 float64 坐标逐位相同。
- **采样**：`default_rng(20260617).choice(eligible, 1000, replace=False)` 需要复现 NumPy 的 SeedSequence、PCG64 与两条抽样分支。4 景都走 Floyd 分支，由板上护栏覆盖；
  eligible 较少时走的尾部洗牌分支没有板上场景触发，只在本机与 NumPy 对照过 9 组参数。
- **kNN 为什么直接用 SciPy 的内核**：PC 编码器的输入是 15 个近邻的**顺序**，而距离并列时返回哪个点取决于建树与查询的具体实现。
  25data/10.22/1 的点云是规则网格重采样的，1,000 个采样点里 939 个在 16 近邻内有距离并列，589 个正好并列在第 15/16 名的取舍处；
  如果并列时改按下标取舍，106 个点的邻域集合、907 个点的邻域顺序会与 Python 版不同（`cpp_knn_ties.json`，其余 3 景没有并列）。
  所以没有用 nanoflann 之类的通用 kd 树，而是把 SciPy 的 C++ 内核直接编进来，4 景邻域偏移逐位相同。
- **GPU 上的匹配**：特征归一化与点积的求和顺序和 torch（`linalg.norm` + cuBLAS bmm）不同，cosine 不保证逐位相同；4 景的匹配行列仍与 Python 版完全一致。
  另外 torch 对"除以 Python 标量"是乘以倒数，C++ 的打分同样这么算，以便与 Python 版在 `DTP+Spatial` 上不出现 1 ulp 的差别。

### 单样本请求延迟

同一批引擎（`*_fp16.plan`）、同一批输入，同一场次内 Python → C++ → C++ → Python 交替跑两轮，warmup 50 + 测 300 次取中位数，下表是两轮均值（`cpp_latency_summary.json`）；
request 口径 = 页锁定内存 H2D + 推理 + D2H + 同步。C++ 与 Python 的输出、eager 与 CUDA Graph 的输出**全部逐位相同**。

| 模型 / batch | Python eager | Python + Graph | C++ eager | C++ + Graph | C++ Graph 相对 Python Graph |
|---|---|---|---|---|---|
| PC / 1 | 0.493 | 0.419 | 0.408 | 0.254 | 1.65x |
| PC / 8 | 0.516 | 0.448 | 0.440 | 0.283 | 1.59x |
| HSI / 1 | 0.507 | 0.452 | 0.424 | 0.282 | 1.60x |
| HSI / 8 | 0.523 | 0.471 | 0.437 | 0.303 | 1.56x |

（单位 ms；表中加速比由未取整的原始测量值计算。）GPU 段（event 口径）两者基本一样（batch 1 的 PC：Python + Graph 0.230ms，C++ + Graph 0.220ms），差别在每次调用的 CPU 下发开销：
用了 CUDA Graph 之后 Python 的 wall 口径仍比 C++ 高约 0.1ms，这就是 C++ 的收益所在。

### 整景：从原始文件到匹配结果

每景 7 种配置，各自独立进程，正轮 P→K0→…→K5、反轮反向；每个进程内 warmup 2 + 重复 10 次取中位数，页缓存不清（文件 IO 是"文件缓存热"的数字）。
P 是 Python 最终配置 E（同一场次重测）；K0 是功能与调度语义对齐 P 的 C++/CUDA baseline，但已经包含完成原生链路所需的 CUDA kernel 重写，因此不是“只换语言/运行时”的纯移植基线。K1～K5 在 K0 基础上每步只改一处（`cpp_e2e_summary.json`）。

| 场景 | P：Python（E） | K0：C++/CUDA baseline | K1：+ 掩膜/统计量多线程 | K2：+ 投影多线程 | **K3：+ 整块读文件** | **K4：+ 复用 PROJ 变换** | K5：+ LAS 与 HSI 读取同时开始（未采纳） | P→K4 |
|---|---|---|---|---|---|---|---|---|
| 24data/10.6/1 | 254.8 / 253.3 | 210.3 / 204.0 | 212.1 / 196.7 | 198.2 / 197.2 | **149.0 / 148.8** | **148.7 / 148.0** | 154.0 / 154.0 | 1.71x |
| 24data/10.6/10 | 267.7 / 266.8 | 187.1 / 188.3 | 183.9 / 184.1 | 194.6 / 176.5 | **124.9 / 125.2** | **124.7 / 123.8** | 133.1 / 133.0 | 2.15x |
| 25data/8.4/1 | 212.8 / 216.9 | 164.2 / 164.9 | 165.0 / 170.1 | 163.6 / 171.6 | **113.6 / 114.0** | **111.9 / 111.5** | 115.1 / 115.5 | 1.90x |
| 25data/10.22/1 | 232.3 / 230.6 | 183.3 / 187.5 | 175.5 / 175.5 | 176.3 / 179.3 | **130.4 / 130.8** | **129.3 / 128.8** | 132.6 / 131.7 | 1.80x |

（单位 ms，正/反两轮。）

- **P → K0（Python reference → 功能/调度语义对齐的 C++/CUDA baseline）**：默认景 254.8 → 210.3ms。K0 已经包含为完成原生链路所需的实现级重写，因此这部分收益不能解释成纯 Python Runtime 开销。主要变化包括融合窗口匹配 kernel，以及标准化 + patch 提取的融合 CUDA 实现；其中匹配微基准从 40.2ms 降至 4.7ms（8.6x，`cpp_matchbench.json`）。
- **K2 → K3：整块读文件是最大的一项**（默认景 198.2 → 149.0ms）。GDAL 的 `RasterIO` 读这个 21.7MB 的 ENVI BSQ 文件要约 55ms，改成按头部信息（BSQ、float32、小端、无头部偏移）整块 `pread` 一次只要约 8ms，读到的立方体逐位相同（`prep-check` 的 `raw_cube`）；
  不满足这些条件的文件仍走 GDAL。
- **K1、K2（掩膜/统计量多线程、投影多线程）收益小且不稳定**：K0 → K2 各场景、各轮次有正有负，波动幅度约 0～20ms（默认景基本持平）。在 K3 之前，关键路径是"读文件 + 前处理 + GPU 上的 HSI 推理"这一串，LAS 段的 CPU 工作藏在 GPU 推理后面，缩短它们只有部分能体现在总耗时上。保留是因为无害。
- **K4（复用 PROJ 变换对象）**：CRS 初始化 31 → 0.1ms，但同样被 GPU 链路盖住，总耗时只差 0.3ms（噪声内）。保留是因为同一进程处理多景时不必每景重建；单景冷启动不受益。
- **K5（LAS 段与 HSI 读取同时开始）没有采纳**：默认景慢约 5ms（148.7 → 154.0ms）。6 个核上同时跑 HSI 读取/统计量、LAS 读取、6 线程投影和后台建树，核被抢占，标准化与掩膜这一段变慢最明显（2.7 → 9.8ms）。
- **K3 之后整景是 GPU 受限的**：默认景 K4 的 148.7ms ≈ 头部约 18ms（读 8.3 + 标准化与掩膜 2.7 + patch 提取与发射 6.5）+ GPU 链路 132.9ms（上传、patch、HSI 推理，PC 推理在第二个 stream 并行；两部分略有重叠）。HSI 推理本身与 Python 版相同（同一个引擎），
  在当前 TensorRT 10.7、输入形状、MAXN_SUPER 功耗模式以及已经尝试的 FP16 / 大 batch / optimization level / TF32 配置下，没有获得进一步稳定且满足护栏的收益。
- **也评估过、没有做的**：只对匹配窗口覆盖到的有效像元做 HSI 推理。4 景里窗口并集覆盖了 99%～100% 的有效像元（1,000 个点的 11×11 窗口在 96×165 左右的影像上几乎铺满，`cpp_window_cover.json`），省不下算力。

### 进程冷启动、内存、能效

默认景，P 与 K4 交替（冷启动各 5 次；能效 ABBA：P、K4、K4、P，各连续跑 30 秒，中间夹 15 秒空闲），`cpp_power_summary.json`：

| | P：Python（E） | K4：C++ |
|---|---|---|
| 进程启动到第一个整景结果 | 4.50 s | **0.65 s** |
| 其中引擎加载 / 第一次整景 | 0.25 s / 0.77 s | 0.22 s / 0.22 s |
| 进程 RSS 峰值 | 2,132 MB | **746 MB** |
| 整板内存峰值（相对空闲，tegrastats） | 1,179 MB | **176 MB** |
| 稳态每景耗时 | 255.0 ms | 149.0 ms |
| 平均整板功率 | 18.0 W | 21.5 W |
| 每景能耗（整板） | 4.60 J | **3.20 J** |
| 每景能耗（高于空闲 7.50W） | 2.69 J | 2.08 J |
| 景 / s / W | 0.217 | 0.313 |

Python 版从进程启动到出结果的 4.50s 里，引擎加载与第一次整景共约 1.02s，其余约 3.5s 是进程启动、Python import（torch、TensorRT 与 polygraphy 绑定、GDAL 等）和 CUDA 初始化，C++ 版没有这一段；
内存差异同理：Python 进程里有 torch 的 CUDA 上下文与缓存分配器。在共享 7.4 GiB 内存的 Orin Nano 上，这个差异比耗时更实际（Python 版在内存紧张时早先出现过一次 NvMap 分配失败，靠及时释放 patch 缓冲解决）。
（整板内存峰值这一项对板上其它驻留状态比较敏感，波动较大；方向和数量级稳定：K4 显著低于 P。）

**散热**：风扇保持系统默认的 nvfancontrol 闭环（没有锁满：锁满会抬高整板功率、污染能效数字），时钟由 jetson_clocks 锁定。C++ 这部分测量期间 tj 最高 64.2°C（延迟 54.3°C、端到端 62.5°C），远低于降频温度，没有降频，风扇转速不影响这些数字。

**C++ 版的边界**：只支持未压缩的 LAS（不支持 LAZ）；HSI 的整块读只在 ENVI BSQ float32 小端时启用，其余走 GDAL；GDAL/PROJ 用的是 conda 环境里的库；没有 INT8（同第 2 节）。

## 6. 能效与冷启动

tegrastats 记录整板输入功率 VDD_IN（每 500ms），每个配置（缓存输入链路）连续跑 25 秒以上，前后各 20 秒空闲作基线（`opt_final2_energy.json`）：

| | 每景耗时 | 平均整板功率 | GPU 利用率 | 每景能耗（整板） | 每景能耗（高于空闲） | 景/s/W |
|---|---|---|---|---|---|---|
| S0 | 2416 ms | 10.26 W | 13% | 24.8 J | 6.78 J | 0.040 |
| **S11b** | **181 ms** | 20.75 W | 93% | **3.76 J** | **2.41 J** | **0.266** |

空闲功率 7.46W。GPU 更忙使功率翻倍，但每景只用 1/13 的时间，**每景能耗降到 1/6.6，每瓦吞吐提高 6.6 倍**。

冷启动（新进程；MAXN_SUPER、时钟锁定；没有用 sudo 清页缓存，所以是“进程冷、文件缓存热”）：

- 缓存输入链路（`opt_final2_coldstart_S11b_*.json`）：CUDA 初始化 0.09s，载入缓存 1.10s，两个引擎反序列化 + 建 context + 分配 buffer 0.16s；第 1 次整景 0.50s，第 2 次起 0.18s。
- 完整端到端部署入口（`e2e_las_summary.json` 中 E 的 `engine_load_sec` / `cold_start_sec`）：引擎加载与 buffer 分配约 0.25s；第 1 次整景 0.69～0.77s（含 CUDA/PROJ 首次初始化），之后稳态 0.21～0.27s（4 景）。

## 7. 没有收益或没做的

| 项 | 结果 |
|---|---|
| 不依赖 torch 的 runner（S3） | 速度持平，保留是因为部署不依赖 torch |
| `builderOptimizationLevel=5` | GPU 时间不变，构建 47s → 257s，不采纳 |
| profile 2048 档 | 只比 1024 快 3%，显存翻倍；在 7.4 GiB 共享内存上第一次构建 PC 2048 时 OOM（`logs/o2_sweep_first_attempt_oom.log`），释放前面的对象后重跑成功 |
| 合并匹配结果的 D2H | 8 次合成 1 次，差异 <0.2ms，不采纳 |
| 标准化与掩膜整体放 GPU（S11）、stem TF32（S10T） | 见第 3 节，未达采纳条件 |
| kNN 查询多核（D） | 只快 1～5ms，保留在最终配置里（无副作用），但不算主要贡献 |
| INT8 | 见第 2 节 |
| 其他功耗模式（15W / 7W） | 未测，需要 sudo 切换、可能要重启板子 |

## 8. 局限

- 文件读取是页缓存命中时的数字；所有数字都在 MAXN_SUPER、时钟锁定下测得。
- 能效用的是 tegrastats 的整板输入功率，时间戳精度 1 秒（统计时去掉每段首尾各 1 秒）。
- TensorRT 版本 10.7（板）与 10.13（x86）不同，两个平台的数字不是同口径，只作为参照并列。
- 多场景只有 4 景（覆盖两个年份、有效像元 6,808～9,645），更大或更稀疏的场景没有测。

## 复现命令（板上）

说明：下面命令中的 `pylibs/` 是板端通过 `pip --target` 安装额外 Python 包时使用的本地依赖目录，不属于 Git 仓库内容，因此不会出现在公开目录树中。

`cpp_e2e_run.sh`、`cpp_power_run.sh`、`cpp_latency_run.sh` 重跑时不删除旧结果：会被覆盖的文件先移进
结果目录旁边的 `<目录>_archive/replaced_at_<时间戳>_<随机后缀>/`，同名文件重复归档时加 `.dupN` 后缀，
不覆盖先归档的那份；只有 `cpp_e2e_run.sh` 的 `tegrastats.log` 按设计跨次追加（`SCENES=` 只重跑部分场景时
不丢其余场景的记录）。

```bash
# 推理与优化实验：系统 Python，PYTHONPATH=pylibs:scripts
python3 scripts/gen_reference.py                    # 板上 FP32 参考（跨平台底噪）
python3 scripts/build_trt.py --modes fp32_notf32 fp32_tf32 fp16
python3 scripts/build_trt.py --which hsi --modes fp16 --precision-policy mixed --fp32-groups stem
python3 scripts/verify_trt.py && python3 scripts/benchmark.py
python3 scripts/jetson_scene_opt.py sweep --tiers 512 1024 2048            # 大 batch 档位扫描
python3 scripts/jetson_scene_opt.py compare --tier 1024                     # S0～S11b 累加对比
python3 scripts/jetson_scene_opt.py --scene-npz <缓存> compare --tier 1024 --configs S0 S11b
python3 scripts/jetson_scene_opt.py latency                                 # CUDA Graph
python3 scripts/jetson_scene_opt.py energy && python3 scripts/jetson_scene_opt.py energy-report --tegrastats <日志> --phases <json>

# 完整端到端：hspc-jetson 环境
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib PYTHONNOUSERSITE=1 PROJ_DATA=$CONDA_PREFIX/share/proj GDAL_DATA=$CONDA_PREFIX/share/gdal HSPC_DATA_ROOT=<原始数据根目录>
python scripts/deploy_jetson.py --scene 24data/10.6/1                       # 最终配置 E
python scripts/deploy_scene.py --scene 24data/10.6/1 --engine-suffix _scene1024 --max-batch 1024   # 对照组 A

# C++ 部署（构建见第 5 节；运行环境变量同上）
./cpp/build/hspc_deploy --scene 24data/10.6/1 --config K4                  # 最终配置；--config K0..K5 是累加对比的各步
./cpp/build/hspc_latency --which pc --engine engines/pc_fp16.plan --samples <样本.f32>   # 单样本延迟（样本由 cpp_guardrail.py latency-refs 导出）
python scripts/cpp_guardrail.py prep-check 24data/10.6/1 <导出目录>          # 逐位护栏：CPU 前处理（hspc_prep_check --dump-dir 导出）
python scripts/cpp_guardrail.py deploy-check 24data/10.6/1 <导出目录>        # 逐位护栏：整景链路（hspc_deploy --dump-dir 导出）
scripts/cpp_latency_run.sh && scripts/cpp_e2e_run.sh && scripts/cpp_power_run.sh   # 板上对比：延迟 / 整景 / 冷启动与能效
python scripts/cpp_latency_summarize.py; python scripts/cpp_e2e_summarize.py; python scripts/cpp_power_summarize.py   # 汇总（本机，读拉回的结果）
```
