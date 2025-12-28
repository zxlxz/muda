#include "cuda/metal.h"
#include "cudart/cuda_runtime_api.h"

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

  auto command_queue = static_cast<CUstream_st*>(stream);
  command_queue->release();
  return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  if (!stream) {
    stream = &CUstream_st::global();
  }

  // use a empty command buffer to synchronize
  auto command_buffer = stream->commandBuffer();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  command_buffer->release();
  return cudaSuccess;
}
