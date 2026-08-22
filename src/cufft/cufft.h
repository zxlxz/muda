#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct cuComplex {
  float x;
  float y;
};

using cufftReal = float;
using cufftComplex = struct cuComplex;

enum cufftResult {
  CUFFT_SUCCESS,
  CUFFT_INVALID_PLAN,
  CUFFT_ALLOC_FAILED,
  CUFFT_INVALID_TYPE,
  CUFFT_INVALID_VALUE,
  CUFFT_INTERNAL_ERROR,
  CUFFT_EXEC_FAILED,
  CUFFT_SETUP_FAILED,
  CUFFT_INVALID_SIZE,
  CUFFT_UNALIGNED_DATA,
  CUFFT_INVALID_DEVICE,
  CUFFT_NO_WORKSPACE,
  CUFFT_NOT_IMPLEMENTED,
  CUFFT_NOT_SUPPORTED,
  CUFFT_MISSING_DEPENDENCY,
  CUFFT_NVRTC_FAILURE,
  CUFFT_NVJITLINK_FAILURE,
  CUFFT_NVSHMEM_FAILURE,
};

enum cufftType {
  CUFFT_R2C = 0x2a,  // real to complex
  CUFFT_C2R = 0x2b,  // complex to real
  CUFFT_C2C = 0x29,  // complex to complex
};

using cufftHandle = int;

static constexpr auto CUFFT_FORWARD = -1;
static constexpr auto CUFFT_INVERSE = 1;
static constexpr auto CUFFT_PLAN_NULL = -1;

// NOTE: only 1d single is supported now

cufftResult cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch);
cufftResult cufftDestroy(cufftHandle plan);

cufftResult cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction);
cufftResult cufftExecR2C(cufftHandle plan, cufftReal* idata, cufftComplex* odata);
cufftResult cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata);

using CUstream = struct CUstream_st*;
cufftResult cufftSetStream(cufftHandle plan, CUstream stream);

#ifdef __cplusplus
}
#endif
