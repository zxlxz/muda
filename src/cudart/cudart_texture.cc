#include "../cuda/metal.h"
#include "cuda_runtime_api.h"

static auto toFilterMode(cudaTextureFilterMode mode) -> MTL::SamplerMinMagFilter {
  switch (mode) {
    case cudaFilterModePoint: return MTL::SamplerMinMagFilterNearest;
    case cudaFilterModeLinear: return MTL::SamplerMinMagFilterLinear;
  }
}

static auto toAddressMode(cudaTextureAddressMode mode) -> MTL::SamplerAddressMode {
  switch (mode) {
    case cudaAddressModeWrap: return MTL::SamplerAddressModeRepeat;
    case cudaAddressModeClamp: return MTL::SamplerAddressModeClampToEdge;
    case cudaAddressModeMirror: return MTL::SamplerAddressModeMirrorRepeat;
    case cudaAddressModeBorder: return MTL::SamplerAddressModeClampToBorderColor;
  }
}

static auto makeSamplerState(const cudaTextureDesc& pTexDesc, MTL::Texture* tex) -> MTL::SamplerState* {
  if (pTexDesc.readMode != cudaReadModeElementType) {
    return nullptr;
  }

  const auto filterMode = toFilterMode(pTexDesc.filterMode);

  const MTL::SamplerAddressMode addressMode[] = {
      toAddressMode(pTexDesc.addressMode[0]),
      toAddressMode(pTexDesc.addressMode[1]),
      toAddressMode(pTexDesc.addressMode[2]),
  };

  auto samplerDesc = AutoRelease{MTL::SamplerDescriptor::alloc()->init()};
  if (!samplerDesc) {
    return nullptr;
  }

  const auto normalizedCoords = pTexDesc.normalizedCoords != 0;

  samplerDesc->setMinFilter(filterMode);
  samplerDesc->setMagFilter(filterMode);
  samplerDesc->setSAddressMode(addressMode[0]);
  samplerDesc->setTAddressMode(addressMode[1]);
  samplerDesc->setRAddressMode(addressMode[2]);
  samplerDesc->setNormalizedCoordinates(normalizedCoords);

  auto& device = CUdevice_st::global();
  auto samplerState = device.newSamplerState(samplerDesc, tex);

  return samplerState;
}

cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObj,
                                    const cudaResourceDesc* pResDesc,
                                    const cudaTextureDesc* pTexDesc,
                                    const cudaResourceViewDesc* /*pResViewDesc*/) {
  if (!pTexObj || !pResDesc || !pTexDesc) {
    return cudaErrorInvalidValue;
  }

  if (pTexDesc->readMode != cudaReadModeElementType) {
    return cudaErrorNotSupported;
  }

  if (pResDesc->resType != cudaResourceTypeArray) {
    return cudaErrorNotSupported;
  }

  auto texture = pResDesc->res.array.array;
  if (!texture) {
    return cudaErrorInvalidDevicePointer;
  }

  auto samplerState = makeSamplerState(*pTexDesc, texture);
  if (!samplerState) {
    return cudaErrorMemoryAllocation;
  }
  *pTexObj = __builtin_bit_cast(cudaTextureObject_t, samplerState);

  return cudaSuccess;
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObj) {
  if (!texObj) {
    return cudaSuccess;
  }

  auto& device = CUdevice_st::global();

  auto sampler = __builtin_bit_cast(MTL::SamplerState*, texObj);
  device.delSamplerState(sampler);

  return cudaSuccess;
}
