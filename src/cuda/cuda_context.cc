#include "cuda.h"
#include "metal.h"

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  (void)pctx;
  (void)flags;
  (void)dev;
  return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext ctx) {
  (void)ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
  (void)pctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
  (void)ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxPushCurrent(CUcontext ctx) {
  (void)ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent(CUcontext* pctx) {
  (void)pctx;
  return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  (void)pctx;
  (void)dev;
  return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
  (void)dev;
  return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize() {
  auto& device = CUdevice_st::global();
  device.Synchronize();
  return CUDA_SUCCESS;
}
