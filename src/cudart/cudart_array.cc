#include "../cuda/cuda.h"
#include "cuda_runtime_api.h"

static auto toArrayFormat(const cudaChannelFormatDesc& desc) -> CUarray_format {
  switch (desc.f) {
    default: break;
    case cudaChannelFormatKindUnsigned: {
      switch (desc.x) {
        case 8: return CU_AD_FORMAT_SIGNED_INT8;
        case 16: return CU_AD_FORMAT_SIGNED_INT16;
        case 32: return CU_AD_FORMAT_SIGNED_INT32;
      }
      break;
    }
    case cudaChannelFormatKindSigned: {
      switch (desc.x) {
        case 8: return CU_AD_FORMAT_UNSIGNED_INT8;
        case 16: return CU_AD_FORMAT_UNSIGNED_INT16;
        case 32: return CU_AD_FORMAT_UNSIGNED_INT32;
      }
      break;
    }
    case cudaChannelFormatKindFloat: {
      switch (desc.x) {
        case 16: return CU_AD_FORMAT_HALF;
        case 32: return CU_AD_FORMAT_FLOAT;
      }
      break;
    }
  }
  return CU_AD_FORMAT_MAX;
}

static auto toChannelFormat(CUarray_format fmt) -> cudaChannelFormatDesc {
  switch (fmt) {
    default: return {0, 0, 0, 0, cudaChannelFormatKindNone};

    case CU_AD_FORMAT_UNSIGNED_INT8: return {8, 0, 0, 0, cudaChannelFormatKindUnsigned};
    case CU_AD_FORMAT_UNSIGNED_INT16: return {16, 0, 0, 0, cudaChannelFormatKindUnsigned};
    case CU_AD_FORMAT_UNSIGNED_INT32: return {32, 0, 0, 0, cudaChannelFormatKindUnsigned};

    case CU_AD_FORMAT_SIGNED_INT8: return {8, 0, 0, 0, cudaChannelFormatKindSigned};
    case CU_AD_FORMAT_SIGNED_INT16: return {16, 0, 0, 0, cudaChannelFormatKindSigned};
    case CU_AD_FORMAT_SIGNED_INT32: return {32, 0, 0, 0, cudaChannelFormatKindSigned};

    case CU_AD_FORMAT_HALF: return {16, 0, 0, 0, cudaChannelFormatKindFloat};
    case CU_AD_FORMAT_FLOAT: return {32, 0, 0, 0, cudaChannelFormatKindFloat};
  }
}

cudaError_t cudaMalloc3DArray(cudaArray_t* array,
                              const cudaChannelFormatDesc* desc,
                              cudaExtent extent,
                              cudaArrayFlags flags) {
  if (!array) {
    return cudaErrorInvalidValue;
  }

  const auto arr_desc = CUDA_ARRAY3D_DESCRIPTOR_st{
      .Width = extent.width,
      .Height = extent.height,
      .Depth = extent.depth,
      .Format = toArrayFormat(*desc),
      .NumChannels = 1,  // Metal textures are single-channel in this implementation
      .Flags = static_cast<unsigned>(flags),
  };

  if (auto err = ::cuArray3DCreate_v2(array, &arr_desc)) {
    return static_cast<cudaError_t>(err);
  }
  return cudaSuccess;
}

cudaError_t cudaFreeArray(cudaArray_t array) {
  if (auto err = ::cuArrayDestroy(array)) {
    return static_cast<cudaError_t>(err);
  }
  return cudaSuccess;
}

cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc* pDesc, cudaExtent* pExtent, int* pFlags, cudaArray_t array) {
  auto t = CUDA_ARRAY3D_DESCRIPTOR_st{};
  if (auto err = ::cuArray3DGetDescriptor_v2(&t, array)) {
    return static_cast<cudaError_t>(err);
  }

  if (pDesc) {
    *pDesc = toChannelFormat(t.Format);
  }

  if (pExtent) {
    pExtent->width = t.Width;
    pExtent->height = t.Height;
    pExtent->depth = t.Depth;
  }

  if (pFlags) {
    *pFlags = t.Flags;
  }

  return cudaSuccess;
}

cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms* p) {
  return ::cudaMemcpy3DAsync(p, nullptr);
}

cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) {
  if (!p) {
    return cudaErrorInvalidValue;
  }

  auto t = CUDA_MEMCPY3D_st{};
  // src
  t.srcMemoryType = CU_MEMORYTYPE_HOST;
  t.srcXInBytes = p->srcPtr.pitch;
  t.srcHost = p->srcPtr.ptr;

  // dst
  t.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  t.dstArray = p->dstArray;

  // size
  t.WidthInBytes = p->extent.width;
  t.Height = p->extent.height;
  t.Depth = p->extent.depth;

  if (auto err = ::cuMemcpy3DAsync_v2(&t, stream)) {
    return static_cast<cudaError_t>(err);
  }

  return cudaErrorNotSupported;
}
