#include <limits.h>
#include <sys/sysctl.h>

#include "cuda.h"
#include "metal.h"

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
  if (!device || ordinal < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (ordinal != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }

  *device = 0;
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
  if (!name || len <= 0 || dev != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  static const auto METAL_DEVICE_NAME = "Metal CUDA Device";
  strncpy(name, METAL_DEVICE_NAME, len - 1);
  name[len - 1] = '\0';

  return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) {
  if (!bytes) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }

  size_t memorySize = 0;
  size_t memorySizeLength = sizeof(memorySize);
  if (sysctlbyname("hw.memsize", &memorySize, &memorySizeLength, nullptr, 0) != 0) {
    return CUDA_ERROR_UNKNOWN;
  }
  *bytes = memorySize;

  return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
  if (!pi) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }

  auto& device = MetalCtx::global();
  const auto maxThreads = device->maxThreadsPerThreadgroup();

  auto val = 0UL;
  switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: val = maxThreads.width * maxThreads.height * maxThreads.depth; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: val = maxThreads.width; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: val = maxThreads.height; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: val = maxThreads.depth; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: val = 2147483647; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: val = 65535; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: val = 65535; break;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: val = device->maxThreadgroupMemoryLength(); break;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: val = 1; break;
    case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT: val = 1; break;
    case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: val = 0; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: val = 3; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: val = 2; break;
    default: return CUDA_ERROR_INVALID_VALUE;
  }

  *pi = static_cast<int>(val);
  return CUDA_SUCCESS;
}
