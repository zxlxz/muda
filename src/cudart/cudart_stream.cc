#include "../cuda/metal.h"
#include "cuda_runtime_api.h"

cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
  if (!pStream) {
    return cudaErrorInvalidValue;
  }

  auto& device = CUdevice_st::global();
  auto command_queue = device.newCommandQueue();
  if (!command_queue) {
    return cudaErrorMemoryAllocation;
  }

  *pStream = static_cast<cudaStream_t>(command_queue);
  return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  if (!stream) {
    return cudaSuccess;
  }

  auto& device = CUdevice_st::global();
  device.delCommandQueue(stream);
  return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  if (!stream) {
    auto& device = CUdevice_st::global();
    stream = static_cast<CUstream_st*>(device.defaultStream());
  }

  // use a empty command buffer to synchronize
  auto command_buffer = AutoRelease{stream->commandBuffer()};
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return cudaSuccess;
}
