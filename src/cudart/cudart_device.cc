#include "../cuda/cuda.h"
#include "cuda_runtime_api.h"

cudaError_t cudaGetDeviceCount(int* pCount) {
  if (!pCount) {
    return cudaErrorInvalidValue;
  }
  if (auto err = ::cuDeviceGetCount(pCount)) {
    return static_cast<cudaError_t>(err);
  }
  return cudaSuccess;
}

cudaError_t cudaGetDevice(int* pDevice) {
  if (!pDevice) {
    return cudaErrorInvalidValue;
  }

  auto dev = CUdevice{};
  if (auto err = ::cuDeviceGet(&dev, 0)) {
    return static_cast<cudaError_t>(err);
  }
  *pDevice = dev;
  return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
  return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
  const auto DEFAULT_MP_COUNT = 10;  // default multiprocessor count

  if (!prop) {
    return cudaErrorInvalidValue;
  }

  prop->multiProcessorCount = DEFAULT_MP_COUNT;
  if (auto err = ::cuDeviceTotalMem_v2(&prop->totalGlobalMem, device)) {
    return static_cast<cudaError_t>(err);
  }

  if (auto err = ::cuDeviceGetName(prop->name, sizeof(prop->name), device)) {
    return static_cast<cudaError_t>(err);
  }

  return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize() {
  if (auto err = ::cuCtxSynchronize()) {
    return static_cast<cudaError_t>(err);
  }
  return cudaSuccess;
}
