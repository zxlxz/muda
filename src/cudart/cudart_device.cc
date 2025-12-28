#include "cuda/metal.h"
#include "cudart/cuda_runtime_api.h"

static constexpr int METAL_DEVICE_ID = 0;

cudaError_t cudaGetDeviceCount(int* count) {
  if (!count) {
    return cudaErrorInvalidValue;
  }
  *count = 1;
  return cudaSuccess;
}

cudaError_t cudaGetDevice(int* device) {
  if (!device) {
    return cudaErrorInvalidValue;
  }
  *device = METAL_DEVICE_ID;
  return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
  if (device != METAL_DEVICE_ID) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
  const auto DEFAULT_MP_COUNT = 10;  // default multiprocessor count

  if (!prop) {
    return cudaErrorInvalidValue;
  }

  if (device != METAL_DEVICE_ID) {
    return cudaErrorInvalidValue;
  }

  auto& impl = CUdevice_st::global();

  const auto name = impl.name()->cString(NS::UTF8StringEncoding);
  strncpy(prop->name, name, sizeof(prop->name) - 1);

  prop->totalGlobalMem = impl.recommendedMaxWorkingSetSize();

  // cannot get real multiprocessor count from metal device,
  // so use the default value
  prop->multiProcessorCount = DEFAULT_MP_COUNT;

  return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize() {
  return cudaStreamSynchronize(nullptr);
}
