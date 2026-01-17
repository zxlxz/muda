#include "cuda.h"
#include "metal.h"

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
  if (!device || ordinal < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto& dev = CUdevice_st::global();
  if (ordinal > 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }

  *device = &dev;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int* count) {
  if (!count) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  *count = 1;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
  if (!name || len <= 0 || !dev) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  static const auto METAL_DEVICE_NAME = "Metal CUDA Device";
  strncpy(name, METAL_DEVICE_NAME, len - 1);
  name[len - 1] = '\0';

  return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
  if (!bytes || !dev) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto& device = CUdevice_st::global();
  *bytes = device.maxBufferLength();

  return CUDA_SUCCESS;
}
