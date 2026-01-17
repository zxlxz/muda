#include "../cuda/metal.h"
#include "cuda_runtime_api.h"

static auto toArrayFlags(MTL::TextureType texType) -> cudaArrayFlags {
  switch (texType) {
    case MTL::TextureType1D: return cudaArrayDefault;
    case MTL::TextureType2D: return cudaArrayDefault;
    case MTL::TextureType3D: return cudaArrayDefault;
    case MTL::TextureType1DArray: return cudaArrayLayered;
    case MTL::TextureType2DArray: return cudaArrayLayered;
    default: return cudaArrayDefault;
  }
}

static auto toChannelFormatDesc(MTL::PixelFormat pixelFormat) -> cudaChannelFormatDesc {
  switch (pixelFormat) {
    default: return {0, 0, 0, 0, cudaChannelFormatKindNone};
    case MTL::PixelFormatR8Sint: return {8, 0, 0, 0, cudaChannelFormatKindSigned};
    case MTL::PixelFormatR16Sint: return {16, 0, 0, 0, cudaChannelFormatKindSigned};
    case MTL::PixelFormatR32Sint: return {32, 0, 0, 0, cudaChannelFormatKindSigned};
    case MTL::PixelFormatR8Uint: return {8, 0, 0, 0, cudaChannelFormatKindUnsigned};
    case MTL::PixelFormatR16Uint: return {16, 0, 0, 0, cudaChannelFormatKindUnsigned};
    case MTL::PixelFormatR32Uint: return {32, 0, 0, 0, cudaChannelFormatKindUnsigned};
    case MTL::PixelFormatR16Float: return {16, 0, 0, 0, cudaChannelFormatKindFloat};
    case MTL::PixelFormatR32Float: return {32, 0, 0, 0, cudaChannelFormatKindFloat};
  }
}

static auto toPixelFormat(const cudaChannelFormatDesc& desc) -> MTL::PixelFormat {
  // only support single channel formats
  if (desc.y != 0 || desc.z != 0 || desc.w != 0) {
    return MTL::PixelFormatUnspecialized;
  }

  if (desc.x <= 0 || desc.x % 8 != 0) {
    return MTL::PixelFormatUnspecialized;
  }

  const auto s = desc.x / 8 - 1;
  if (s < 0 || s >= 3) {
    return MTL::PixelFormatUnspecialized;
  }

  static const MTL::PixelFormat U[] = {
      MTL::PixelFormatR8Uint,
      MTL::PixelFormatR16Uint,
      MTL::PixelFormatR32Uint,
  };
  static const MTL::PixelFormat I[] = {
      MTL::PixelFormatR8Sint,
      MTL::PixelFormatR16Sint,
      MTL::PixelFormatR32Sint,
  };
  static const MTL::PixelFormat F[] = {
      MTL::PixelFormatUnspecialized,
      MTL::PixelFormatR16Float,
      MTL::PixelFormatR32Float,
  };

  switch (desc.f) {
    case cudaChannelFormatKindSigned: return I[s];
    case cudaChannelFormatKindUnsigned: return U[s];
    case cudaChannelFormatKindFloat: return F[s];
    default: return MTL::PixelFormatUnspecialized;
  }
}

static auto toPixelType(const cudaExtent& extent, cudaArrayFlags flags) -> MTL::TextureType {
  switch (flags) {
    default:
    case cudaArrayDefault: {
      return extent.depth > 1 ? MTL::TextureType3D : extent.height > 1 ? MTL::TextureType2D : MTL::TextureType1D;
    }
    case cudaArrayLayered: {
      return extent.depth > 1 ? MTL::TextureType2DArray : MTL::TextureType1DArray;
    }
  }
}

static auto makeTextureDesc(const cudaChannelFormatDesc& desc, const cudaExtent& extent, cudaArrayFlags flags)
    -> MTL::TextureDescriptor* {
  const auto texFormat = toPixelFormat(desc);
  if (texFormat == MTL::PixelFormatUnspecialized) {
    return nullptr;
  }

  const auto texType = toPixelType(extent, flags);
  auto texDesc = MTL::TextureDescriptor::alloc()->init();
  texDesc->setTextureType(texType);
  texDesc->setStorageMode(MTL::StorageModeShared);
  texDesc->setPixelFormat(texFormat);
  texDesc->setWidth(extent.width);
  texDesc->setHeight(extent.height);

  if (flags == cudaArrayLayered) {
    texDesc->setArrayLength(extent.depth);
  } else {
    texDesc->setDepth(extent.depth);
  }
  texDesc->setMipmapLevelCount(1);

  return texDesc;
}

cudaError_t cudaMalloc3DArray(cudaArray_t* array,
                              const cudaChannelFormatDesc* desc,
                              cudaExtent extent,
                              cudaArrayFlags flags) {
  if (!array) {
    return cudaErrorInvalidValue;
  }

  auto texDesc = AutoRelease{makeTextureDesc(*desc, extent, flags)};
  if (!texDesc) {
    return cudaErrorNotSupported;
  }

  auto& device = CUdevice_st::global();
  auto texture = device.newTexture(texDesc);

  *array = static_cast<cudaArray_t>(texture);
  return cudaSuccess;
}

cudaError_t cudaFreeArray(cudaArray_t array) {
  if (!array) {
    return cudaSuccess;
  }

  auto& device = CUdevice_st::global();
  device.delTexture(array);
  return cudaSuccess;
}

cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms* p) {
  return cudaMemcpy3DAsync(p, nullptr);
}

cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* pExtent, int* pFlags, cudaArray_t array) {
  if (!array) {
    return cudaErrorInvalidValue;
  }

  if (desc) {
    const auto pixelFormat = array->pixelFormat();
    *desc = toChannelFormatDesc(pixelFormat);
  }

  if (pExtent) {
    const auto w = array->width();
    const auto h = array->height();
    const auto d = array->depth();
    *pExtent = {.width = w, .height = h, .depth = d};
  }

  const auto textureType = array->textureType();
  if (pFlags) {
    *pFlags = toArrayFlags(textureType);
  }

  return cudaSuccess;
}

cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) {
  if (!p) {
    return cudaErrorInvalidValue;
  }

  if (p->extent.width == 0 || p->extent.height == 0 || p->extent.depth == 0) {
    return cudaErrorInvalidValue;
  }

  // copy from host ptr to device array
  if (p->srcPtr.ptr && p->dstArray) {
    const auto srcPtr = p->srcPtr.ptr;
    const auto dstBuffer = p->dstArray;

    // copy from srcPtr to dstBuffer
    const auto bytesPerRow = p->srcPtr.pitch;
    const auto bytesPerImage = bytesPerRow * p->extent.height;
    const auto region = MTL::Region::Make3D(0, 0, 0, p->extent.width, p->extent.height, p->extent.depth);
    dstBuffer->replaceRegion(region, 0, 0, srcPtr, bytesPerRow, bytesPerImage);
    return cudaSuccess;
  }

  return cudaErrorNotSupported;
}
