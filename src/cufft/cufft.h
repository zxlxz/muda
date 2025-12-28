#pragma once

struct cuComplex {
  float x;
  float y;
};

using cufftReal = float;
using cufftComplex = struct cuComplex;

enum cufftResult {
  CUFFT_SUCCESS = 0,
  CUFFT_INVALID_PLAN = 1,
  CUFFT_ALLOC_FAILED = 2,
  CUFFT_INVALID_TYPE = 3,
  CUFFT_INVALID_VALUE = 4,
  CUFFT_INTERNAL_ERROR = 5,
  CUFFT_EXEC_FAILED = 6,
  CUFFT_SETUP_FAILED = 7,
  CUFFT_INVALID_SIZE = 8,
  CUFFT_UNALIGNED_DATA = 9,
};

enum cufftType {
  CUFFT_R2C = 0x2a,  // real to complex
  CUFFT_C2R = 0x2b,  // complex to real
  CUFFT_C2C = 0x29,  // complex to complex
};

using cufftHandle = unsigned int;
static constexpr auto CUFFT_FORWARD = -1;
static constexpr auto CUFFT_INVERSE = 1;
static constexpr auto CUFFT_PLAN_NULL = static_cast<cufftHandle>(-1);

// NOTE: only 1d single is supported now
cufftResult cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch);

cufftResult cufftDestroy(cufftHandle plan);

cufftResult cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction);
cufftResult cufftExecR2C(cufftHandle plan, cufftReal* idata, cufftComplex* odata);
cufftResult cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata);
