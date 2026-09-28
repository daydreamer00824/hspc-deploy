cmake_minimum_required(VERSION 3.18)
project(hspc_jetson_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_ARCHITECTURES 87)  # Orin (sm_87)

# CPU 端必须关闭 FMA 融合：aarch64 上 GCC 默认把 a*b+c 融成一条 FMA，
# 而 NumPy / laspy 的 X*scale+offset 是两步舍入，融合后结果不再逐位相同。
# nvcc 也不开 --use_fast_math：标准化里的除法要 IEEE 舍入。
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffp-contract=off -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fmad=false -Xcompiler=-ffp-contract=off")

find_package(CUDAToolkit REQUIRED)
find_library(NVINFER nvinfer HINTS /usr/lib/aarch64-linux-gnu REQUIRED)

# ---- 只依赖 TensorRT / CUDA：单样本延迟（不需要 GDAL/PROJ）
add_library(trt_engine STATIC src/trt_engine.cpp)
target_include_directories(trt_engine PUBLIC src /usr/include/aarch64-linux-gnu)
target_link_libraries(trt_engine PUBLIC ${NVINFER} CUDA::cudart)

add_executable(hspc_latency src/latency_main.cpp)
target_link_libraries(hspc_latency PRIVATE trt_engine)

# ---- CPU 前处理（GDAL / PROJ 来自 conda 环境，与 Python 参考实现用的是同一份库）
set(HSPC_CONDA_PREFIX "$ENV{HOME}/miniforge3/envs/hspc-jetson" CACHE PATH "conda 环境前缀（提供 GDAL / PROJ）")
find_library(GDAL_LIB gdal HINTS ${HSPC_CONDA_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
find_library(PROJ_LIB proj HINTS ${HSPC_CONDA_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
find_package(Threads REQUIRED)

set_source_files_properties(third_party/scipy_ckdtree/build.cxx third_party/scipy_ckdtree/query.cxx PROPERTIES COMPILE_OPTIONS "-w")  # 第三方代码不带我们的告警选项
add_library(prep STATIC
  src/hsi_io.cpp src/hsi_prep.cpp src/np_random.cpp src/las_pipeline.cpp
  third_party/scipy_ckdtree/build.cxx third_party/scipy_ckdtree/query.cxx)
target_include_directories(prep PUBLIC src third_party/scipy_ckdtree ${HSPC_CONDA_PREFIX}/include)
# 链接期也要用 conda 的较新 libstdc++（GDAL 引用了 GLIBCXX_3.4.32），否则系统自带的旧版会让链接失败
find_library(CONDA_STDCXX stdc++ HINTS ${HSPC_CONDA_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
target_link_libraries(prep PUBLIC ${CONDA_STDCXX} ${GDAL_LIB} ${PROJ_LIB} Threads::Threads)
target_link_options(prep PUBLIC -Wl,-rpath-link,${HSPC_CONDA_PREFIX}/lib)
# 运行时用 conda 里较新的 libstdc++（GDAL 需要）；TensorRT 库对更新的 libstdc++ 向后兼容
set(CMAKE_BUILD_RPATH ${HSPC_CONDA_PREFIX}/lib)

add_executable(hspc_prep_check src/prep_check_main.cpp)
target_link_libraries(hspc_prep_check PRIVATE prep)

# ---- 整景部署程序（CUDA kernel + TensorRT + GDAL/PROJ）
# 匹配打分规则是课题方法本身，放在不入库的 src/matching_rule.cuh 里（同 scripts/matching.py）。
# 公开仓库没有该文件：只跳过依赖它的 hspc_deploy，不影响 hspc_latency / hspc_prep_check。
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/matching_rule.cuh)
  add_executable(hspc_deploy src/deploy_main.cpp src/deploy_pipeline.cpp src/kernels.cu)
  target_link_libraries(hspc_deploy PRIVATE prep trt_engine)
else()
  message(WARNING
    "src/matching_rule.cuh is not included; hspc_deploy will not be built. "
    "hspc_latency and hspc_prep_check remain available.")
endif()
