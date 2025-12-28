# MUDA - CUDA on Metal

<div align="center">

![MUDA](https://img.shields.io/badge/MUDA-CUDA%20on%20Metal-blue?style=for-the-badge&logo=apple)
![Platform](https://img.shields.io/badge/Platform-macOS%2012.0+-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A CUDA-compatible API implementation for macOS using Apple's Metal framework.**

</div>

---

## 📋 Overview

MUDA provides a CUDA-like programming interface that runs on macOS by leveraging Apple's Metal framework. This project enables CUDA applications to be ported to macOS with minimal code changes, bridging the gap between NVIDIA CUDA and Apple's Metal compute capabilities.

### 🎯 Goals

- ✅ Provide CUDA runtime API compatibility layer
- ✅ Enable CUDA code migration to macOS/Metal
- ✅ Support FFT operations via Apple's vDSP
- ✅ Maintain familiar CUDA programming patterns

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Your Application                      │
│                  (CUDA-compatible code)                      │
├─────────────────────────────────────────────────────────────┤
│                     CUDA Runtime API                         │
│           (cudart - CUDA Runtime Library)                   │
│         cuModuleLoad, cuLaunchKernel, etc.                  │
├─────────────────────────────────────────────────────────────┤
│                     CUDA Driver API                          │
│            (cuda - Driver Interface)                        │
│         cuModuleLoad, cuModuleGetFunction, etc.            │
├─────────────────────────────────────────────────────────────┤
│                      metal-cpp                              │
│          (Apple's Metal C++ Wrapper)                        │
├─────────────────────────────────────────────────────────────┤
│                         Metal                                │
│           (Apple's GPU Computing Framework)                 │
├─────────────────────────────────────────────────────────────┤
│                    GPU Hardware                              │
│              (Apple Silicon / Intel Mac)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
muda/
├── README.md                 # This file
├── CLAUDE.md                 # Development instructions
│
├── metal-cpp/               # Apple's Metal C++ wrapper (DO NOT MODIFY)
│   ├── Metal/
│   ├── MetalKit/
│   └── Foundation/
│
├── src/
│   ├── cuda/                # CUDA Driver API implementation
│   │   ├── cuda.h           # CUresult, CUmodule, etc.
│   │   ├── cuda.cc          # cuModuleLoad, cuLaunchKernel, etc.
│   │   └── metal.h          # Metal integration
│   │
│   ├── cudart/              # CUDA Runtime API implementation
│   │   ├── cuda_runtime_api.h
│   │   └── cuda_runtime_api.cc
│   │
│   └── cufft/               # CUDA FFT Library (using Apple vDSP)
│       ├── cufft.h
│       └── cufft.cc
│
└── include/
    ├── cuda/                # Public CUDA headers
    ├── cudart/              # Public cudart headers
    └── cufft/               # Public cufft headers
```

---

## ✨ Features

### CUDA Driver API (`cuda.h`)

| Function | Status | Description |
|----------|--------|-------------|
| `cuModuleLoad` | ✅ Complete | Load a CUDA module (Metal library) |
| `cuModuleUnload` | ✅ Complete | Unload a CUDA module |
| `cuModuleGetFunction` | ✅ Complete | Get function handle from module |
| `cuLaunchKernel` | ✅ Complete | Launch a compute kernel |
| More functions | 🔄 Planned | Ongoing development |

### CUDA Runtime API (`cuda_runtime_api.h`)

| Function | Status | Description |
|----------|--------|-------------|
| `cudaGetDeviceCount` | ✅ Complete | Get number of available devices |
| `cudaGetDevice` | ✅ Complete | Get current device ordinal |
| `cudaSetDevice` | ✅ Complete | Set current device |
| `cudaMalloc` | ✅ Complete | Allocate device memory |
| `cudaFree` | ✅ Complete | Free device memory |
| `cudaMemcpy` | ✅ Complete | Memory copy operations |
| `cudaStreamCreate` | ✅ Complete | Create a stream |
| More functions | 🔄 Planned | Ongoing development |

### CUDA FFT (`cufft.h`)

| Function | Status | Description |
|----------|--------|-------------|
| `cufftPlan1d` | ✅ Complete | Create 1D FFT plan |
| `cufftExecC2C` | ✅ Complete | Complex-to-complex transform |
| `cufftExecC2R` | ✅ Complete | Complex-to-real transform |
| `cufftExecR2C` | ✅ Complete | Real-to-complex transform |
| `cufftDestroy` | ✅ Complete | Destroy FFT plan |

---

## 🚀 Getting Started

### Prerequisites

- **macOS 12.0+** (Monterey or later)
- **Xcode 14.0+** with command-line tools
- **Apple Silicon** (M1/M2/M3) or **Intel Mac with Metal**

### Building

```bash
# Clone the repository
git clone https://github.com/yourusername/muda.git
cd muda

# configure
cmake --build build

# build
cmake -B build
```

### Basic Usage

```cpp
#include <cuda.h>
#include <cuda_runtime_api.h>

int main() {
    // Initialize CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        return CUDA_ERROR_NO_DEVICE;
    }

    // Set device
    cudaSetDevice(0);

    // Allocate memory
    float* d_data;
    cudaMalloc(&d_data, sizeof(float) * 1024);

    // Load and launch kernel
    CUmodule module;
    cuModuleLoad(&module, "mykernel.metallib");

    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, "myKernel");

    cuLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0, nullptr, nullptr, nullptr);

    // Cleanup
    cudaFree(d_data);
    cuModuleUnload(module);

    return cudaSuccess;
}
```

---

## 🔧 Metal Shaders

MUDA requires Metal shader libraries (`.metallib` files). Compile your shaders using `metal`:

```bash
# Compile .metal to .metallib
xcrun metal myshader.metal -o myshader.air
```

### Shader Example

```metal
#include <metal_stdlib>
using namespace metal;

[[kernel]] void myKernel(device float* data [[buffer(0)]],
                     uint id [[thread_position_in_grid]]) {
    data[id] = data[id] * 2.0f;
}
```

---

## 📊 Limitations

- ⚠️ **NVIDIA-specific features** (Tensor Cores, cuBLAS, cuDNN) are not available
- ⚠️ **PTX assembly** is not supported; use Metal Shading Language
- ⚠️ Some advanced CUDA features may have different semantics
- ⚠️ Performance characteristics differ from NVIDIA GPUs

---

## 🆚 CUDA vs Metal Error Mapping

| CUDA Error | Metal Error |
|------------|-------------|
| `CUDA_ERROR_INVALID_PTX` | `MTLLibraryErrorCompileFailure` |
| `CUDA_ERROR_INVALID_IMAGE` | `MTLLibraryErrorUnsupported` |
| `CUDA_ERROR_FILE_NOT_FOUND` | `MTLLibraryErrorFileNotFound` |
| `CUDA_ERROR_NOT_FOUND` | `MTLLibraryErrorFunctionNotFound` |

---

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

### Development Notes

1. **Do NOT modify `metal-cpp/`** - This is Apple's official wrapper
2. Follow the existing code style
3. Add tests for new functionality
4. Update documentation accordingly

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Apple** for Metal and metal-cpp
- **NVIDIA** for the CUDA API specification
- **Apple's Accelerate framework** for vDSP FFT implementation

---

<div align="center">

**Made with ❤️ for the macOS GPU computing community**

</div>
