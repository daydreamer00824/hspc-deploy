# Environment and Dependencies

## 1. Benchmark environment

The reported measurements were collected with Python 3.10, TensorRT 10.13.3.9
(CUDA 12.9), PyTorch 2.13.0 (CUDA 13.0 for the FP32 baseline), ONNX opset 17,
and an NVIDIA RTX 3060 12GB GPU (SM 8.6) under WSL2.

TensorRT Python bindings and CUDA are system- and GPU-environment-dependent
dependencies; they are not represented as a portable pip-only installation.

## 2. Core inference and benchmark dependencies

- numpy
- torch
- onnx
- onnxruntime
- tensorrt
- polygraphy
- modelopt (used by the optional explicit-QDQ PTQ script)

## 3. Whole-scene preprocessing dependencies

- scipy
- GDAL / `osgeo.gdal`
- pyproj
- laspy

## 4. README asset generation

- matplotlib

## 5. Reproducibility scope

This public repository excludes the private model definitions, trained weights,
source and calibration data, and the private cross-modal matching rule. This
document describes the software stack used by the public deployment engineering
code; it does not imply that the repository can reproduce the original research
pipeline end to end without those private materials.
