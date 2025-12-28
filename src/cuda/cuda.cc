#include "cuda/cuda.h"

#include "cuda/metal.h"

CUresult cuModuleLoad(CUmodule* module, const char* path) {
  if (!module) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!path) {
    return CUDA_ERROR_FILE_NOT_FOUND;
  }

  const auto u8_path = NS::String::string(path, NS::UTF8StringEncoding);
  auto& device = CUdevice_st::global();

  auto err = static_cast<NS::Error*>(nullptr);
  auto library = device.newLibrary(u8_path, &err);
  if (err) {
    const auto domain = err->domain()->utf8String();
    if (strcmp(domain, "MTLLibraryErrorDomain") == 0) {
      switch (err->code()) {
        case MTL::LibraryErrorUnsupported: return CUresult::CUDA_ERROR_INVALID_IMAGE;
        case MTL::LibraryErrorInternal: return CUresult::CUDA_ERROR_INVALID_CONTEXT;
        case MTL::LibraryErrorCompileFailure: return CUresult::CUDA_ERROR_INVALID_SOURCE;
        case MTL::LibraryErrorCompileWarning: return CUresult::CUDA_ERROR_INVALID_SOURCE;
        case MTL::LibraryErrorFunctionNotFound: return CUresult::CUDA_ERROR_NOT_FOUND;
        case MTL::LibraryErrorFileNotFound: return CUresult::CUDA_ERROR_FILE_NOT_FOUND;
      }
    }
    return CUDA_ERROR_INVALID_IMAGE;
  }

  if (!library) {
    return CUDA_ERROR_UNKNOWN;
  }

  *module = static_cast<CUmod_st*>(library);
  return CUDA_SUCCESS;
}

CUresult cuModuleUnload(CUmodule hmod) {
  if (!hmod) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  hmod->release();
  return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
  if (!hfunc || !hmod || !name) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  const auto u8_name = NS::String::string(name, NS::UTF8StringEncoding);
  const auto func = hmod->newFunction(u8_name);
  if (!func) {
    return CUDA_ERROR_NOT_FOUND;
  }

  *hfunc = static_cast<CUfunction>(func);
  return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(CUfunction f,
                        uint32_t gridDimX,
                        uint32_t gridDimY,
                        uint32_t gridDimZ,
                        uint32_t blockDimX,
                        uint32_t blockDimY,
                        uint32_t blockDimZ,
                        uint32_t /*sharedMemBytes*/,
                        CUstream stream,
                        const CUParam params[],
                        void** /*extra*/) {
  if (!f) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  stream = stream ? stream : &CUstream_st::global();

  // Get device and create compute pipeline state
  auto& device = CUdevice_st::global();
  NS::Error* error = nullptr;
  auto pipeline_state = device.newComputePipelineState(f, &error);
  if (!pipeline_state) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Create command buffer and compute encoder
  auto command_buffer = stream->commandBuffer();
  if (!command_buffer) {
    pipeline_state->release();
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto compute_encoder = command_buffer->computeCommandEncoder();
  if (!compute_encoder) {
    command_buffer->release();
    pipeline_state->release();
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Set compute pipeline state
  compute_encoder->setComputePipelineState(pipeline_state);

  // set kernel parameters
  for (auto i = 0U;; ++i) {
    if (params == nullptr || params[i]._type == CUParam::None) {
      break;
    }
    const auto& p = params[i];
    switch (p._type) {
      case CUParam::None: break;
      case CUParam::Bytes: compute_encoder->setBytes(p._data, p._size, i); break;
      case CUParam::Buffer:
        if (auto buff = device.findBuffer(p._data)) {
          compute_encoder->setBuffer(buff.buffer, buff.offset, i);
        }
        break;
      case CUParam::Texture: compute_encoder->setTexture(static_cast<const MTL::Texture*>(p._data), i); break;
      case CUParam::Sampler: compute_encoder->setSamplerState(static_cast<const MTL::SamplerState*>(p._data), i); break;
    }
  }

  // Set threadgroups and threads per threadgroup
  const auto threadsPerGrid = MTL::Size{gridDimX * blockDimX, gridDimY * blockDimY, gridDimZ * blockDimZ};
  const auto threadsPerThreadGroup = MTL::Size{blockDimX, blockDimY, blockDimZ};
  compute_encoder->dispatchThreads(threadsPerGrid, threadsPerThreadGroup);
  compute_encoder->endEncoding();

  // Commit and wait for completion
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  // Cleanup
  compute_encoder->release();
  command_buffer->release();
  pipeline_state->release();

  return CUDA_SUCCESS;
}
