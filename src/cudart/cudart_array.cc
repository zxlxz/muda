#include "cuda/metal.h"
#include "cudart/cuda_runtime_api.h"

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
  struct ChannelInfo {
    MTL::PixelFormat format;
    cudaChannelFormatKind kind;
    int bits;
  };

  static const ChannelInfo channelInfos[] = {
      {MTL::PixelFormatR8Sint, cudaChannelFormatKindSigned, 8},
      {MTL::PixelFormatR16Sint, cudaChannelFormatKindSigned, 16},
      {MTL::PixelFormatR32Sint, cudaChannelFormatKindSigned, 32},
      {MTL::PixelFormatR8Uint, cudaChannelFormatKindUnsigned, 8},
      {MTL::PixelFormatR16Uint, cudaChannelFormatKindUnsigned, 16},
      {MTL::PixelFormatR32Uint, cudaChannelFormatKindUnsigned, 32},
      {MTL::PixelFormatR16Float, cudaChannelFormatKindFloat, 16},
      {MTL::PixelFormatR32Float, cudaChannelFormatKindFloat, 32},
  };

  auto desc = cudaChannelFormatDesc{0, 0, 0, 0, cudaChannelFormatKindNone};
  for (const auto& info : channelInfos) {
    if (info.format == pixelFormat) {
      desc.x = info.bits;
      desc.f = info.kind;
      break;
    }
  }

  return desc;
}

static auto makePixelFormat(const cudaChannelFormatDesc& desc) -> MTL::PixelFormat {
  MTL::PixelFormat pixelFormat = MTL::PixelFormatUnspecialized;

  // only support single channel formats
  if (desc.y != 0 || desc.z != 0 || desc.w != 0) {
    return pixelFormat;
  }

  const auto sx = desc.x;
  switch (desc.f) {
    case cudaChannelFormatKindSigned:
      return sx == 8    ? MTL::PixelFormatR8Sint   // i8
             : sx == 16 ? MTL::PixelFormatR16Sint  // i16
             : sx == 32 ? MTL::PixelFormatR32Sint  // i32
                        : MTL::PixelFormatUnspecialized;
    case cudaChannelFormatKindUnsigned:
      return sx == 8    ? MTL::PixelFormatR8Uint   // u8
             : sx == 16 ? MTL::PixelFormatR16Uint  // u16
             : sx == 32 ? MTL::PixelFormatR32Uint  // u32
                        : MTL::PixelFormatUnspecialized;
    case cudaChannelFormatKindFloat:
      return sx == 16   ? MTL::PixelFormatR16Float  // f16
             : sx == 32 ? MTL::PixelFormatR32Float  // f32
                        : MTL::PixelFormatUnspecialized;
    default: return MTL::PixelFormatUnspecialized;
  }
}

static auto makePixelType(const cudaExtent& extent, cudaArrayFlags flags) -> MTL::TextureType {
  const auto ndim = (extent.depth > 1) ? 3 : ((extent.height > 1) ? 2 : 1);
  switch (ndim) {
    case 1: return MTL::TextureType1D;
    case 2: return flags == cudaArrayDefault ? MTL::TextureType2D : MTL::TextureType1DArray;
    case 3: return flags == cudaArrayDefault ? MTL::TextureType3D : MTL::TextureType2DArray;
  }
  return MTL::TextureType3D;
}

static auto makeTextureDesc(const cudaChannelFormatDesc& desc, const cudaExtent& extent, cudaArrayFlags flags)
    -> MTL::TextureDescriptor* {
  const auto texFormat = makePixelFormat(desc);
  if (texFormat == MTL::PixelFormatUnspecialized) {
    return nullptr;
  }

  const auto texType = makePixelType(extent, flags);

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

  auto texDesc = makeTextureDesc(*desc, extent, flags);
  if (!texDesc) {
    return cudaErrorNotSupported;
  }

  auto& device = CUdevice_st::global();
  auto texture = device.newTexture(texDesc);
  texDesc->release();

  *array = static_cast<cudaArray*>(texture);
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
    const auto cudaExtent = ::cudaExtent{
        .width = array->width(),
        .height = array->height(),
        .depth = array->depth(),
    };
    *pExtent = cudaExtent;
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
    const auto dstBuffer = static_cast<MTL::Texture*>(p->dstArray);

    // copy from srcPtr to dstBuffer
    const auto bytesPerRow = p->srcPtr.pitch;
    const auto bytesPerImage = bytesPerRow * p->extent.height;
    const auto region = MTL::Region::Make3D(0, 0, 0, p->extent.width, p->extent.height, p->extent.depth);
    dstBuffer->replaceRegion(region, 0, 0, srcPtr, bytesPerRow, bytesPerImage);
    return cudaSuccess;
  }

  return cudaErrorNotSupported;
}
