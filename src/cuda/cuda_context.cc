#include "cuda.h"
#include "metal.h"

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  return CUDA_SUCCESS;  
}

CUresult cuCtxDestroy(CUcontext ctx) {
  return CUDA_SUCCESS;  
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
  return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
  return CUDA_SUCCESS;
}

CUresult cuCtxPushCurrent(CUcontext ctx) {
  return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent(CUcontext* pctx) {
  return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize() {
  auto& device = CUdevice_st::global();
  device.Synchronize();
  return CUDA_SUCCESS;
}
