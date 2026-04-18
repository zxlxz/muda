#include "../cuda/cuda.h"
#include "cuda_runtime_api.h"

cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
  if (auto err = ::cuStreamCreate(pStream, 0)) {
    return static_cast<cudaError_t>(err);
  }
  return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  if (auto err = ::cuStreamDestroy_v2(stream)) {
    return static_cast<cudaError_t>(err);
  }
  return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  if (auto err = ::cuStreamSynchronize(stream)) {
    return static_cast<cudaError_t>(err);
  }
  return cudaSuccess;
}
