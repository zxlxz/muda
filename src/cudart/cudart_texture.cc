#include "../cuda/cuda.h"
#include "cuda_runtime_api.h"

cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObj,
                                    const cudaResourceDesc* pResDesc,
                                    const cudaTextureDesc* pTexDesc,
                                    const cudaResourceViewDesc* /*pResViewDesc*/) {
  if (!pTexObj || !pResDesc || !pTexDesc) {
    return cudaErrorInvalidValue;
  }

  auto resDesc = CUDA_RESOURCE_DESC_st{};
  resDesc.resType = static_cast<CUresourcetype>(pResDesc->resType);
  resDesc.res.array.hArray = pResDesc->res.array.array;

  auto texDesc = CUDA_TEXTURE_DESC_st{};
  texDesc.addressMode[0] = static_cast<CUaddress_mode>(pTexDesc->addressMode[0]);
  texDesc.addressMode[1] = static_cast<CUaddress_mode>(pTexDesc->addressMode[1]);
  texDesc.addressMode[2] = static_cast<CUaddress_mode>(pTexDesc->addressMode[2]);
  texDesc.filterMode = static_cast<CUfilter_mode>(pTexDesc->filterMode);
  texDesc.flags = static_cast<CUaddress_mode>(pTexDesc->normalizedCoords);

  if (auto err = ::cuTexObjectCreate(pTexObj, &resDesc, &texDesc, nullptr)) {
    return static_cast<cudaError_t>(err);
  }

  return cudaSuccess;
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObj) {
  if (auto err = ::cuTexObjectDestroy(texObj)) {
    return static_cast<cudaError_t>(err);
  }
  return cudaSuccess;
}
