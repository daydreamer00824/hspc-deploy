# Environment and Dependencies

## 1. x86 benchmark environment

The reported measurements were collected with Python 3.10, TensorRT 10.13.3.9
(CUDA 12.9), PyTorch 2.13.0 (CUDA 13.0 for the FP32 baseline), ONNX opset 17,
and an NVIDIA RTX 3060 12GB GPU (SM 8.6) under WSL2.

TensorRT Python bindings and CUDA are system- and GPU-environment-dependent
dependencies; they are not represented as a portable pip-only installation.

## 2. Jetson benchmark and C++ environment

The Jetson measurements were collected on a Jetson Orin Nano Engineering
Reference Developer Kit Super with 7.4 GiB shared memory, L4T R36.4.7
(JetPack 6), CUDA 12.6, TensorRT 10.7.0, Python 3.10, and NVIDIA's Jetson build
of PyTorch 2.5.0. Whole-scene preprocessing used GDAL 3.11.4, PROJ 9.6.2,
laspy 2.7.0, and SciPy 1.15.2. Exact setup and measurement details are in
[the Jetson deployment report](jetson.md).

The C++ programs require CMake 3.18 or newer, a C++17 compiler, nvcc, the CUDA
and TensorRT C++ headers and libraries, and GDAL/PROJ. The public tree can build
`hspc_latency` and `hspc_prep_check`; `hspc_deploy` additionally requires the
private `cpp/src/matching_rule.cuh` described in the README.

## 3. Core inference and benchmark dependencies

- numpy
- torch
- onnx
- onnxruntime
- tensorrt
- polygraphy
- modelopt (used by the optional explicit-QDQ PTQ script)

## 4. Whole-scene preprocessing dependencies

- scipy
- GDAL / `osgeo.gdal`
- pyproj
- laspy

## 5. README asset generation

- matplotlib

## 6. Reproducibility scope

This public repository excludes the private model definitions, trained weights,
source and calibration data, and the private cross-modal matching rule. This
document describes the software stack used by the public deployment engineering
code; it does not imply that the repository can reproduce the original research
pipeline end to end without those private materials.
