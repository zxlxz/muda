#include "cuda.h"
#include "metal.h"

CUresult cuStreamCreate(CUstream* phStream, unsigned int flags) {
  if (!phStream) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto& device = CUdevice_st::global();
  auto command_queue = device.newCommandQueue();
  if (!command_queue) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  *phStream = static_cast<CUstream>(command_queue);
  return CUDA_SUCCESS;
}

CUresult cuStreamDestroy(CUstream hStream) {
  if (!hStream) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto& device = CUdevice_st::global();
  device.delCommandQueue(hStream);
  return CUDA_SUCCESS;
}

CUresult cuStreamSynchronize(CUstream hStream) {
  auto command_queue = static_cast<MTL::CommandQueue*>(hStream);

  if (!command_queue) {
    auto& device = CUdevice_st::global();
    command_queue = device.defaultStream();
  }

  // use a empty command buffer to synchronize
  auto command_buffer = AutoRelease{command_queue->commandBuffer()};
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return CUDA_SUCCESS;
}
