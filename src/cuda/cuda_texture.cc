#include "cuda.h"
#include "metal.h"

static auto toSamplerFilter(CUfilter_mode mode) -> MTL::SamplerMinMagFilter {
  switch (mode) {
    default:
    case CU_TR_FILTER_MODE_POINT: return MTL::SamplerMinMagFilterNearest;
    case CU_TR_FILTER_MODE_LINEAR: return MTL::SamplerMinMagFilterLinear;
  }
}

static auto toSamplerAddressMode(CUaddress_mode mode) -> MTL::SamplerAddressMode {
  switch (mode) {
    case CU_TR_ADDRESS_MODE_WRAP: return MTL::SamplerAddressModeRepeat;
    default:
    case CU_TR_ADDRESS_MODE_CLAMP: return MTL::SamplerAddressModeClampToEdge;
    case CU_TR_ADDRESS_MODE_MIRROR: return MTL::SamplerAddressModeMirrorRepeat;
    case CU_TR_ADDRESS_MODE_BORDER: return MTL::SamplerAddressModeClampToBorderColor;
  }
}

static auto makeSamplerState(const CUDA_TEXTURE_DESC& pTexDesc, const MTL::Texture& tex) -> MTL::SamplerState* {
  auto samplerDesc = AutoRelease{MTL::SamplerDescriptor::alloc()};
  if (!samplerDesc) {
    return nullptr;
  }
  if (!samplerDesc->init()) {
    return nullptr;
  }

  const auto filterMode = toSamplerFilter(pTexDesc.filterMode);
  const MTL::SamplerAddressMode addressMode[] = {
      toSamplerAddressMode(pTexDesc.addressMode[0]),
      toSamplerAddressMode(pTexDesc.addressMode[1]),
      toSamplerAddressMode(pTexDesc.addressMode[2]),
  };

  const auto normalizedCoords = bool(pTexDesc.flags & CU_TRSF_NORMALIZED_COORDINATES);

  samplerDesc->setMinFilter(filterMode);
  samplerDesc->setMagFilter(filterMode);
  samplerDesc->setSAddressMode(addressMode[0]);
  samplerDesc->setTAddressMode(addressMode[1]);
  samplerDesc->setRAddressMode(addressMode[2]);
  samplerDesc->setNormalizedCoordinates(normalizedCoords);

  auto& device = CUdevice_st::global();
  auto samplerState = device.newSamplerState(samplerDesc, const_cast<MTL::Texture*>(&tex));
  return samplerState;
}

CUresult cuTexObjectCreate(CUtexObject* pTexObject,
                           const CUDA_RESOURCE_DESC* pResDesc,
                           const CUDA_TEXTURE_DESC* pTexDesc,
                           const CUDA_RESOURCE_VIEW_DESC* pResViewDesc) {
  if (!pTexObject || !pResDesc || !pTexDesc) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (pResDesc->resType != CU_RESOURCE_TYPE_ARRAY) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  auto texture = pResDesc->res.array.array;
  if (!texture) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto samplerState = makeSamplerState(*pTexDesc, *texture);
  if (!samplerState) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  *pTexObject = __builtin_bit_cast(CUtexObject, samplerState);

  return CUDA_SUCCESS;
}

CUresult cuTexObjectDestroy(CUtexObject texObject) {
  if (texObject == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto samplerState = __builtin_bit_cast(MTL::SamplerState*, texObject);
  auto& device = CUdevice_st::global();
  device.delSamplerState(samplerState);

  return CUDA_SUCCESS;
}
