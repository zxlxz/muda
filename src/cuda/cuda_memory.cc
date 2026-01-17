#include "cuda.h"
#include "metal.h"

CUresult cuMemGetInfo(size_t* free, size_t* total) {
  if (!free || !total) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto& device = CUdevice_st::global();
  const auto maxSize = device.maxBufferLength();
  const auto usedSize = device.currentAllocatedSize();

  *total = maxSize;
  *free = maxSize - usedSize;

  return CUDA_SUCCESS;
}

CUresult cuMemAlloc(void** dptr, size_t bytesize) {
  if (!dptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto& device = CUdevice_st::global();
  auto buffer = device.newBuffer(bytesize, MTL::ResourceStorageModeShared);
  if (!buffer) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  *dptr = buffer->contents();
  return CUDA_SUCCESS;
}

CUresult cuMemFree(void* dptr) {
  if (!dptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto& device = CUdevice_st::global();

  // find the buffer info
  auto bufRange = device.findBuffer(dptr);
  if (bufRange.buffer == nullptr || bufRange.offset != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  device.delBuffer(bufRange.buffer);

  return CUDA_SUCCESS;
}

CUresult cuMemAllocManaged(void** dptr, size_t bytesize, unsigned int flags) {
  // Note: Metal does not have a separate managed memory model.
  // We treat managed memory as shared memory.
  return cuMemAlloc(dptr, bytesize);
}

CUresult cuMemAllocHost(void** hptr, size_t bytesize) {
  if (!hptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  *hptr = malloc(bytesize);
  if (!*hptr) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  return CUDA_SUCCESS;
}

CUresult cuMemFreeHost(void* p) {
  if (!p) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  free(p);
  return CUDA_SUCCESS;
}

CUresult cuMemcpy(void* dst, const void* src, size_t bytesize) {
  if (!dst || !src || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::memcpy(dst, src, bytesize);
  return CUDA_SUCCESS;
}

CUresult cuMemcpyAsync(void* dst, const void* src, size_t bytesize, CUstream hStream) {
  (void)hStream;

  if (!dst || !src || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  // In Metal's shared memory model, we can directly memcpy
  // and stream is ignored
  std::memcpy(dst, src, bytesize);
  return CUDA_SUCCESS;
}

CUresult cuMemPrefetchAsync(void* devPtr, size_t count, CUmemLocation location, unsigned int flags, CUstream hStream) {
  (void)devPtr;
  (void)count;
  (void)location;
  (void)flags;
  (void)hStream;
  return CUDA_SUCCESS;
}

CUresult cuMemsetD8(void* dst, unsigned char uc, size_t N) {
  return cuMemsetD8Async(dst, uc, N, nullptr);
}

CUresult cuMemsetD8Async(void* dst, unsigned char uc, size_t N, CUstream hStream) {
  (void)hStream;

  if (!dst) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (N == 0) {
    return CUDA_SUCCESS;
  }

  // in metal's shared memory model, we can directly memset
  // and stream is ignored
  ::memset(dst, uc, N);
  return CUDA_SUCCESS;
}

CUresult cuMemsetD16(void* dst, unsigned short us, size_t N) {
  return cuMemsetD16Async(dst, us, N, nullptr);
}

CUresult cuMemsetD16Async(void* dst, unsigned short us, size_t N, CUstream hStream) {
  (void)hStream;

  if (!dst) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (N == 0) {
    return CUDA_SUCCESS;
  }

  // in metal's shared memory model, we can directly memset
  // and stream is ignored
  auto ptr = static_cast<unsigned short*>(dst);
  for (size_t i = 0; i < N; ++i) {
    ptr[i] = us;
  }
  return CUDA_SUCCESS;
}

CUresult cuMemsetD32(void* dst, unsigned int ui, size_t N) {
  return cuMemsetD32Async(dst, ui, N, nullptr);
}

CUresult cuMemsetD32Async(void* dst, unsigned int ui, size_t N, CUstream hStream) {
  (void)hStream;

  if (!dst) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (N == 0) {
    return CUDA_SUCCESS;
  }

  // in metal's shared memory model, we can directly memset
  // and stream is ignored
  auto ptr = static_cast<unsigned int*>(dst);
  for (size_t i = 0; i < N; ++i) {
    ptr[i] = ui;
  }
  return CUDA_SUCCESS;
}
