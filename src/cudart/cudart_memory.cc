#include "../cuda/cuda.h"
#include "cuda_runtime_api.h"

cudaError_t cudaMalloc(void** ptr, size_t size) {
  if (!ptr) {
    return cudaErrorInvalidValue;
  }

  auto dptr = CUdeviceptr{};
  if (auto err = ::cuMemAlloc_v2(&dptr, size)) {
    return static_cast<cudaError_t>(err);
  }
  *ptr = reinterpret_cast<void*>(dptr);

  return cudaSuccess;
}

cudaError_t cudaMallocManaged(void** ptr, size_t size, unsigned int flags) {
  (void)flags;
  return ::cudaMalloc(ptr, size);
}

cudaError_t cudaFree(void* ptr) {
  if (ptr == nullptr) {
    return cudaErrorInvalidValue;
  }

  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  if (auto err = ::cuMemFree_v2(dptr)) {
    return static_cast<cudaError_t>(err);
  }
  return cudaSuccess;
}

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
  if (!ptr) {
    return cudaErrorInvalidValue;
  }

  if (auto err = ::cuMemHostAlloc(ptr, size, flags)) {
    return static_cast<cudaError_t>(err);
  }

  return cudaSuccess;
}

cudaError_t cudaFreeHost(void* ptr) {
  if (ptr == nullptr) {
    return cudaErrorInvalidValue;
  }

  if (auto err = ::cuMemFreeHost(ptr)) {
    return static_cast<cudaError_t>(err);
  }
  return cudaSuccess;
}

cudaError_t cudaMemPrefetchAsync(const void* /*ptr*/, size_t /*count*/, cudaMemLocation /*location*/, unsigned int /*flags*/, cudaStream_t /*stream*/) {
  // in metal's shared memory model, prefetch is a no-op
  return cudaSuccess;
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
  return ::cudaMemsetAsync(devPtr, value, count, nullptr);
}

cudaError_t cudaMemsetAsync(void* ptr, int value, size_t count, cudaStream_t stream) {
  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  const auto val = static_cast<unsigned char>(value & 0xFF);
  if (auto err = ::cuMemsetD8Async(dptr, val, count, stream)) {
    return static_cast<cudaError_t>(err);
  }
  return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  return ::cudaMemcpyAsync(dst, src, count, kind, nullptr);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
  const auto d_dst = reinterpret_cast<CUdeviceptr>(dst);
  const auto d_src = reinterpret_cast<CUdeviceptr>(src);

  CUresult err;
  switch (kind) {
    case cudaMemcpyHostToHost:
      err = ::cuMemcpyAsync(d_dst, d_src, count, stream);
      break;
    case cudaMemcpyHostToDevice:
      err = ::cuMemcpyHtoDAsync(d_dst, src, count, stream);
      break;
    case cudaMemcpyDeviceToHost:
      err = ::cuMemcpyDtoHAsync(dst, d_src, count, stream);
      break;
    case cudaMemcpyDeviceToDevice:
      err = ::cuMemcpyDtoDAsync(d_dst, d_src, count, stream);
      break;
    case cudaMemcpyDefault:
      // direction is inferred from the pointers
      err = ::cuMemcpyAsync(d_dst, d_src, count, stream);
      break;
    default: return cudaErrorInvalidMemcpyDirection;
  }

  if (err) {
    return static_cast<cudaError_t>(err);
  }
  return cudaSuccess;
}

cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) {
  if (!attributes || !ptr) {
    return cudaErrorInvalidValue;
  }

  (void)ptr;

  attributes->device = 0;                        // default device
  attributes->type = cudaMemLocationTypeDevice;  // default to device memory

  return cudaSuccess;
}
