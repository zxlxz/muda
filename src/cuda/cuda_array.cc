#include "cuda.h"
#include "metal.h"

static auto toArrayFormat(MTL::PixelFormat pixelFormat) -> CUarray_format {
  switch (pixelFormat) {
    case MTL::PixelFormatR8Uint: return CU_AD_FORMAT_UNSIGNED_INT8;
    case MTL::PixelFormatR16Uint: return CU_AD_FORMAT_UNSIGNED_INT16;
    case MTL::PixelFormatR32Uint: return CU_AD_FORMAT_UNSIGNED_INT32;
    case MTL::PixelFormatR8Sint: return CU_AD_FORMAT_SIGNED_INT8;
    case MTL::PixelFormatR16Sint: return CU_AD_FORMAT_SIGNED_INT16;
    case MTL::PixelFormatR32Sint: return CU_AD_FORMAT_SIGNED_INT32;
    case MTL::PixelFormatR16Float: return CU_AD_FORMAT_HALF;
    case MTL::PixelFormatR32Float: return CU_AD_FORMAT_FLOAT;
    default: return CU_AD_FORMAT_UNSIGNED_INT8;
  }
}

static auto toPixelFormat(CUarray_format format) -> MTL::PixelFormat {
  switch (format) {
    case CU_AD_FORMAT_UNSIGNED_INT8: return MTL::PixelFormatR8Uint;
    case CU_AD_FORMAT_UNSIGNED_INT16: return MTL::PixelFormatR16Uint;
    case CU_AD_FORMAT_UNSIGNED_INT32: return MTL::PixelFormatR32Uint;
    case CU_AD_FORMAT_SIGNED_INT8: return MTL::PixelFormatR8Sint;
    case CU_AD_FORMAT_SIGNED_INT16: return MTL::PixelFormatR16Sint;
    case CU_AD_FORMAT_SIGNED_INT32: return MTL::PixelFormatR32Sint;
    case CU_AD_FORMAT_HALF: return MTL::PixelFormatR16Float;
    case CU_AD_FORMAT_FLOAT: return MTL::PixelFormatR32Float;
    default: return MTL::PixelFormatUnspecialized;
  }
}

static auto toTextureType(const CUDA_ARRAY3D_DESCRIPTOR& desc) -> MTL::TextureType {
  switch (desc.Flags) {
    case CU_ARRAY_LAYERED: {
      return desc.Depth > 1 ? MTL::TextureType2DArray : MTL::TextureType1DArray;
    }
    default: {
      return desc.Depth > 1 ? MTL::TextureType3D : desc.Height > 1 ? MTL::TextureType2D : MTL::TextureType1D;
    }
  }
}

static auto makeTextureDesc(const CUDA_ARRAY3D_DESCRIPTOR& pAllocateArray) -> MTL::TextureDescriptor* {
  const auto texFormat = toPixelFormat(pAllocateArray.Format);
  if (texFormat == MTL::PixelFormatUnspecialized) {
    return nullptr;
  }

  const auto texType = toTextureType(pAllocateArray);
  auto texDesc = MTL::TextureDescriptor::alloc()->init();
  texDesc->setTextureType(texType);
  texDesc->setStorageMode(MTL::StorageModeShared);
  texDesc->setPixelFormat(texFormat);
  texDesc->setWidth(pAllocateArray.Width);
  texDesc->setHeight(pAllocateArray.Height);

  if (pAllocateArray.Flags == CU_ARRAY_LAYERED) {
    texDesc->setArrayLength(pAllocateArray.Depth);
  } else {
    texDesc->setDepth(pAllocateArray.Depth);
  }
  texDesc->setMipmapLevelCount(1);
  return texDesc;
}

CUresult cuArray3DCreate(CUarray* pHandle, const struct CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray) {
  if (!pHandle || !pAllocateArray) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto texDesc = AutoRelease{makeTextureDesc(*pAllocateArray)};
  if (!texDesc) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  auto& device = CUdevice_st::global();
  auto texture = device.newTexture(texDesc);
  *pHandle = static_cast<CUarray>(texture);
  return CUDA_SUCCESS;
}

CUresult cuArrayDestroy(CUarray hArray) {
  if (!hArray) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto& device = CUdevice_st::global();
  device.delTexture(static_cast<MTL::Texture*>(hArray));

  return CUDA_SUCCESS;
}

CUresult cuArray3DGetDescriptor(struct CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray) {
  if (!pArrayDescriptor || !hArray) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  const auto pixelFormat = hArray->pixelFormat();
  const auto textureType = hArray->textureType();
  const auto isLayered = textureType == MTL::TextureType2DArray || textureType == MTL::TextureType1DArray;

  pArrayDescriptor->Width = hArray->width();
  pArrayDescriptor->Height = hArray->height();
  pArrayDescriptor->Depth = hArray->depth();
  pArrayDescriptor->Format = toArrayFormat(pixelFormat);
  pArrayDescriptor->NumChannels = 1;  // Metal textures are single-channel in this implementation
  pArrayDescriptor->Flags = isLayered ? CU_ARRAY_LAYERED : CU_ARRAY_DEFAULT;

  return CUDA_SUCCESS;
}

CUresult cuMemcpy3D(CUDA_MEMCPY3D* pCopy) {
  return cuMemcpy3DAsync(pCopy, nullptr);
}

CUresult cuMemcpy3DAsync(struct CUDA_MEMCPY3D* pCopy, CUstream hStream) {
  (void)hStream;

  if (!pCopy) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  const auto widthInBytes = pCopy->WidthInBytes;
  const auto height = pCopy->Height;
  const auto depth = pCopy->Depth;
  if (widthInBytes == 0 || height == 0 || depth == 0) {
    return CUDA_SUCCESS;
  }

  const auto srcPtr = pCopy->srcHost ? pCopy->srcHost : pCopy->srcDevice;
  const auto dstPtr = pCopy->dstHost ? pCopy->dstHost : pCopy->dstDevice;
  const auto srcArray = pCopy->srcArray;
  const auto dstArray = pCopy->dstArray;
  if (bool(srcPtr) == bool(srcArray)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (bool(dstPtr) == bool(dstArray)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (srcPtr && dstArray) {
    const auto bytesPerRow = pCopy->srcPitch;
    const auto bytesPerImage = bytesPerRow * height;
    const auto region = MTL::Region::Make3D(0, 0, 0, widthInBytes, height, depth);
    dstArray->replaceRegion(region, 0, 0, srcPtr, bytesPerRow, bytesPerImage);
    return CUDA_SUCCESS;
  }

  return CUDA_ERROR_NOT_SUPPORTED;
}
