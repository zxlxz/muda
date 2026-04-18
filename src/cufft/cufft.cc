#include "cufft.h"

#include <Accelerate/Accelerate.h>

#include <mutex>
#include <vector>

struct cufftPlan {
  cufftPlan() = default;
  virtual ~cufftPlan() = default;

  virtual auto exec_c2c(cufftComplex* idata, cufftComplex* odata, int direction) const -> cufftResult = 0;
  virtual auto exec_r2c(cufftReal* Ir, cufftComplex* Oc) const -> cufftResult = 0;
  virtual auto exec_c2r(cufftComplex* Ic, cufftReal* Or) const -> cufftResult = 0;
};

class cufftPlan1D : public cufftPlan {
  cufftType _type = CUFFT_C2C;
  int _len = 0;
  int _batch = 0;

  vDSP_DFT_Interleaved_Setup _fwd = nullptr;
  vDSP_DFT_Interleaved_Setup _inv = nullptr;
  vDSP_DFT_Setup _r2c = nullptr;
  vDSP_DFT_Setup _c2r = nullptr;

 public:
  cufftPlan1D(cufftType type, int len, int batch) : _type{type}, _len{len}, _batch{batch} {
    const auto c2c = vDSP_DFT_Interleaved_ComplextoComplex;
    switch (type) {
      case CUFFT_R2C: {
        _r2c = vDSP_DFT_zrop_CreateSetup(nullptr, _len, vDSP_DFT_FORWARD);
        break;
      }
      case CUFFT_C2R: {
        _c2r = vDSP_DFT_zrop_CreateSetup(nullptr, _len, vDSP_DFT_INVERSE);
        break;
      }
      case CUFFT_C2C: {
        _fwd = vDSP_DFT_Interleaved_CreateSetup(nullptr, _len, vDSP_DFT_FORWARD, c2c);
        _inv = vDSP_DFT_Interleaved_CreateSetup(nullptr, _len, vDSP_DFT_INVERSE, c2c);
        break;
      }
      default: break;
    }
  }

  ~cufftPlan1D() {
    if (_r2c) vDSP_DFT_DestroySetup(_r2c);
    if (_c2r) vDSP_DFT_DestroySetup(_c2r);
    if (_fwd) vDSP_DFT_Interleaved_DestroySetup(_fwd);
    if (_inv) vDSP_DFT_Interleaved_DestroySetup(_inv);
  }

  auto exec_c2c(cufftComplex* idata, cufftComplex* odata, int direction) const -> cufftResult {
    const auto setup = direction == CUFFT_FORWARD ? _fwd : _inv;
    if (!setup) {
      return CUFFT_INVALID_PLAN;
    }

    for (auto i = 0; i < _batch; ++i) {
      const auto ip = reinterpret_cast<DSPComplex*>(idata + i * _len);
      const auto op = reinterpret_cast<DSPComplex*>(odata + i * _len);
      vDSP_DFT_Interleaved_Execute(setup, ip, op);
    }
    return CUFFT_SUCCESS;
  }

  // Ir: fft_len
  // Oc: comp_len = fft_len / 2 + 1
  auto exec_r2c(cufftReal* Ir, cufftComplex* Oc) const -> cufftResult {
    if (!_r2c) {
      return CUFFT_INVALID_PLAN;
    }

    const auto fft_len = _len;
    const auto comp_len = fft_len / 2 + 1;

    auto Ii = static_cast<float*>(::malloc(fft_len * sizeof(float)));
    auto Or = static_cast<float*>(::malloc(comp_len * sizeof(float)));
    auto Oi = static_cast<float*>(::malloc(comp_len * sizeof(float)));

    for (auto i = 0; i < _batch; ++i) {
      vDSP_DFT_Execute(_r2c, Ir + i * fft_len, Ii, Or, Oi);
      this->pack_complex(Or, Oi, Oc + i * comp_len);
    }

    ::free(Ii);
    ::free(Or);
    ::free(Oi);
    return CUFFT_SUCCESS;
  }

  // Ic: comp_len = fft_len / 2 + 1
  // Or: fft_len
  auto exec_c2r(cufftComplex* Ic, cufftReal* Or) const -> cufftResult {
    if (!_c2r) {
      return CUFFT_INVALID_PLAN;
    }

    const auto fft_len = _len;
    const auto comp_len = fft_len / 2 + 1;

    auto Ir = static_cast<float*>(::malloc(comp_len * sizeof(float)));
    auto Ii = static_cast<float*>(::malloc(comp_len * sizeof(float)));
    auto Oi = static_cast<float*>(::malloc(fft_len * sizeof(float)));

    for (auto i = 0; i < _batch; ++i) {
      this->unpack_complex(Ic + i * comp_len, Ir, Ii);
      vDSP_DFT_Execute(_c2r, Ir, Ii, Or, Oi);
    }

    ::free(Ir);
    ::free(Ii);
    ::free(Oi);
    return CUFFT_SUCCESS;
  }

 private:
  // real: _len/2
  // imag: _len/2
  // comp: _len/2 + 1
  void pack_complex(const float* real, const float* imag, cufftComplex* comp) const {
    const auto n = _len / 2;
    comp[0] = {real[0], 0};  // DC component
    comp[n] = {imag[0], 0};  // Nyquist frequency
    for (auto i = 1; i < n; ++i) {
      comp[i] = {real[i], imag[i]};
    }
  }

  // comp: _len/2 + 1
  // real: _len/2
  // imag: _len/2
  void unpack_complex(const cufftComplex* comp, float* real, float* imag) const {
    const auto n = _len / 2;
    real[0] = comp[0].x;  // DC component
    imag[0] = comp[n].x;  // Nyquist frequency
    for (auto i = 1; i < n; ++i) {
      real[i] = comp[i].x;
      imag[i] = comp[i].y;
    }
  }
};

cufftResult cufftPlan1d(cufftHandle* pPlan, int nx, cufftType type, int batch) {
  if (!pPlan || nx <= 0 || batch < 0) {
    return CUFFT_INVALID_VALUE;
  }

  if (batch <= 0) {
    return CUFFT_INVALID_VALUE;
  }

  *pPlan = new cufftPlan1D{type, nx, batch};
  return CUFFT_SUCCESS;
}

cufftResult cufftDestroy(cufftHandle plan) {
  if (!plan) {
    return CUFFT_INVALID_PLAN;
  }

  delete plan;
  return CUFFT_SUCCESS;
}

cufftResult cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction) {
  if (!plan) {
    return CUFFT_INVALID_PLAN;
  }
  if (!idata || !odata) {
    return CUFFT_INVALID_VALUE;
  }

  return plan->exec_c2c(idata, odata, direction);
}

cufftResult cufftExecR2C(cufftHandle plan, cufftReal* ireal, cufftComplex* ocomp) {
  if (!plan) {
    return CUFFT_INVALID_PLAN;
  }
  if (!ireal || !ocomp) {
    return CUFFT_INVALID_VALUE;
  }

  return plan->exec_r2c(ireal, ocomp);
}

cufftResult cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata) {
  if (!plan) {
    return CUFFT_INVALID_PLAN;
  }
  if (!idata || !odata) {
    return CUFFT_INVALID_VALUE;
  }
  return plan->exec_c2r(idata, odata);
}

cufftResult cufftSetStream(cufftHandle plan, CUstream stream) {
  if (!plan) {
    return CUFFT_INVALID_PLAN;
  }
  (void)stream;  // stream is not supported in this implementation
  return CUFFT_SUCCESS;
}
