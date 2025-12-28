#include "cufft.h"

#include <Accelerate/Accelerate.h>

#include <mutex>
#include <vector>

struct FFTInfo {
  cufftType type = CUFFT_C2C;
  unsigned long len = 0;
  int batch = 0;
  vDSP_DFT_Setup r2c = nullptr;
  vDSP_DFT_Setup c2r = nullptr;
  vDSP_DFT_Interleaved_Setup c2c_forward = nullptr;
  vDSP_DFT_Interleaved_Setup c2c_inverse = nullptr;
};

struct FFTPlanCache {
  mutable std::mutex _mutex = {};
  std::vector<FFTInfo> _plans = {};

 public:
  static auto instance() -> FFTPlanCache& {
    static auto cache = FFTPlanCache{};
    return cache;
  }

  auto size() const -> size_t {
    auto lock = std::unique_lock{_mutex};
    return _plans.size();
  }

  auto at(cufftHandle plan) -> FFTInfo* {
    auto lock = std::unique_lock{_mutex};
    if (plan >= _plans.size()) {
      return nullptr;
    }
    return &_plans[plan];
  }

  auto regist(FFTInfo info) -> cufftHandle {
    auto lock = std::unique_lock{_mutex};
    for (auto i = 0U; i < _plans.size(); ++i) {
      if (_plans[i].len == 0) {
        _plans[i] = info;
        return i;
      }
    }
    _plans.push_back(info);
    return static_cast<cufftHandle>(_plans.size() - 1);
  }
};

cufftResult cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch) {
  if (!plan || nx <= 0 || batch < 0) {
    return CUFFT_INVALID_VALUE;
  }

  auto info = FFTInfo{
      .type = type,
      .len = static_cast<unsigned long>(nx),
      .batch = batch == 0 ? 1 : batch,
  };

  const auto r2c = vDSP_DFT_Interleaved_ComplextoComplex;
  switch (type) {
    case CUFFT_R2C:
      info.r2c = vDSP_DFT_zrop_CreateSetup(nullptr, info.len, vDSP_DFT_FORWARD);
      if (!info.r2c) {
        return CUFFT_ALLOC_FAILED;
      }
      break;
    case CUFFT_C2R:
      info.c2r = vDSP_DFT_zrop_CreateSetup(nullptr, info.len, vDSP_DFT_INVERSE);
      if (!info.c2r) {
        return CUFFT_ALLOC_FAILED;
      }
      break;
    case CUFFT_C2C:
      info.c2c_forward = vDSP_DFT_Interleaved_CreateSetup(nullptr, info.len, vDSP_DFT_FORWARD, r2c);
      if (!info.c2c_forward) {
        return CUFFT_ALLOC_FAILED;
      }
      info.c2c_inverse = vDSP_DFT_Interleaved_CreateSetup(nullptr, info.len, vDSP_DFT_INVERSE, r2c);
      if (!info.c2c_inverse) {
        vDSP_DFT_Interleaved_DestroySetup(info.c2c_forward);
        return CUFFT_ALLOC_FAILED;
      }
      break;
    default: return CUFFT_INVALID_TYPE;
  }

  const auto handle = FFTPlanCache::instance().regist(info);
  *plan = handle;
  return CUFFT_SUCCESS;
}

cufftResult cufftDestroy(cufftHandle plan) {
  auto* info = FFTPlanCache::instance().at(plan);
  if (info == nullptr) {
    return CUFFT_INVALID_PLAN;
  }

  if (info->c2r) vDSP_DFT_DestroySetup(info->c2r);
  if (info->r2c) vDSP_DFT_DestroySetup(info->r2c);
  if (info->c2c_forward) vDSP_DFT_Interleaved_DestroySetup(info->c2c_forward);
  if (info->c2c_inverse) vDSP_DFT_Interleaved_DestroySetup(info->c2c_inverse);
  *info = {};
  return CUFFT_SUCCESS;
}

cufftResult cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction) {
  if (!idata || !odata) {
    return CUFFT_INVALID_VALUE;
  }

  const auto* p = FFTPlanCache::instance().at(plan);
  if (p == nullptr) {
    return CUFFT_INVALID_PLAN;
  }

  const auto setup = direction == CUFFT_FORWARD ? p->c2c_forward : p->c2c_inverse;
  if (!setup) {
    return CUFFT_INVALID_PLAN;
  }

  for (auto i = 0U; i < p->batch; ++i) {
    const auto ip = reinterpret_cast<DSPComplex*>(idata + i * p->len);
    const auto op = reinterpret_cast<DSPComplex*>(odata + i * p->len);
    vDSP_DFT_Interleaved_Execute(setup, ip, op);
  }

  return CUFFT_SUCCESS;
}

cufftResult cufftExecR2C(cufftHandle plan, cufftReal* ireal, cufftComplex* ocomp) {
  if (!ireal || !ocomp) {
    return CUFFT_INVALID_VALUE;
  }

  const auto* p = FFTPlanCache::instance().at(plan);
  if (p == nullptr) {
    return CUFFT_INVALID_PLAN;
  }

  const auto setup = p->r2c;
  if (!setup) {
    return CUFFT_INVALID_PLAN;
  }

  const auto fft_len = p->len;
  const auto half_len = fft_len / 2 + 1;
  auto II = std::make_unique<float[]>(fft_len);  // not used
  auto OR = std::make_unique<float[]>(half_len);
  auto OI = std::make_unique<float[]>(half_len);

  const auto Ii = II.get();
  const auto Or = OR.get();
  const auto Oi = OI.get();

  auto vdsp_to_cufft = [&](const float* vr, const float* vi, cufftComplex* c) {
    c[0] = {vr[0], 0};            // DC component
    c[fft_len / 2] = {vi[0], 0};  // Nyquist frequency
    for (auto j = 1U; j < fft_len / 2; ++j) {
      c[j] = {vr[j], vi[j]};
    }
  };

  for (auto i = 0U; i < p->batch; ++i) {
    const auto Ir = ireal + i * fft_len;
    vDSP_DFT_Execute(setup, Ir, Ii, Or, Oi);

    const auto Oc = ocomp + i * half_len;
    vdsp_to_cufft(Or, Oi, Oc);
  }

  return CUFFT_SUCCESS;
}

cufftResult cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata) {
  if (!idata || !odata) {
    return CUFFT_INVALID_VALUE;
  }

  const auto* p = FFTPlanCache::instance().at(plan);
  if (p == nullptr) {
    return CUFFT_INVALID_PLAN;
  }

  const auto setup = p->c2r;
  if (!setup) {
    return CUFFT_INVALID_PLAN;
  }
  const auto fft_len = p->len;
  const auto half_len = fft_len / 2 + 1;

  auto IR = std::make_unique<float[]>(half_len);
  auto II = std::make_unique<float[]>(half_len);
  auto OI = std::make_unique<float[]>(fft_len);  // not used
  const auto Ir = IR.get();
  const auto Ii = II.get();
  const auto Oi = OI.get();

  auto cufft_to_vdsp = [&](const cufftComplex* c, float* vr, float* vi) {
    vr[0] = c[0].x;            // DC component
    vi[0] = c[fft_len / 2].x;  // Nyquist frequency
    for (auto j = 1U; j < fft_len / 2; ++j) {
      vr[j] = c[j].x;
      vi[j] = c[j].y;
    }
  };

  for (auto i = 0U; i < p->batch; ++i) {
    const auto Ic = idata + i * half_len;
    cufft_to_vdsp(Ic, Ir, Ii);

    const auto Or = odata + i * fft_len;
    vDSP_DFT_Execute(setup, Ir, Ii, Or, Oi);
  }
  return CUFFT_SUCCESS;
}
