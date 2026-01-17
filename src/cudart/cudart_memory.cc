#include "../cuda/metal.h"
#include "cuda_runtime_api.h"

cudaError_t cudaMalloc(void** ptr, size_t size) {
  if (!ptr) {
    return cudaErrorInvalidValue;
  }

  if (size == 0) {
    *ptr = nullptr;
    return cudaSuccess;
  }

  auto& device = CUdevice_st::global();
  auto buffer = device.newBuffer(size, MTL::ResourceStorageModeShared);
  if (!buffer) {
    return cudaErrorMemoryAllocation;
  }

  // get the user space address of the buffer
  auto data = buffer->contents();
  *ptr = data;
  return cudaSuccess;
}

cudaError_t cudaFree(void* ptr) {
  if (!ptr) {
    return cudaSuccess;
  }

  auto& device = CUdevice_st::global();

  // find the buffer info
  auto bufferInfo = device.findBuffer(ptr);

  // if buffer not found
  //  means this is not a valid device pointer
  if (bufferInfo.buffer == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }
  device.delBuffer(bufferInfo.buffer);
  return cudaSuccess;
}

cudaError_t cudaMallocManaged(void** ptr, size_t size) {
  return ::cudaMalloc(ptr, size);
}

cudaError_t cudaMemPrefetchAsync(const void* /*ptr*/,
                                 size_t /*count*/,
                                 cudaMemLocation /*location*/,
                                 unsigned int /*flags*/,
                                 cudaStream_t /*stream*/) {
  // in metal's shared memory model, prefetch is a no-op
  return cudaSuccess;
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
  return cudaMemsetAsync(devPtr, value, count, nullptr);
}

cudaError_t cudaMemsetAsync(void* ptr, int value, size_t count, cudaStream_t /*stream*/) {
  if (!ptr) {
    return cudaErrorInvalidValue;
  }

  if (count == 0) {
    return cudaSuccess;
  }

  ::memset(ptr, value, count);
  return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  return cudaMemcpyAsync(dst, src, count, kind, nullptr);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
  (void)stream;

  if (!dst || !src) {
    return cudaErrorInvalidValue;
  }

  if (count == 0) {
    return cudaSuccess;
  }

  // because metal buffer is shared memory, we can directly memcpy
  ::memcpy(dst, src, count);
  return cudaSuccess;
}
