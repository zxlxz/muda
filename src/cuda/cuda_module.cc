#include "cuda.h"
#include "metal.h"

static auto toCudaError(NS::Error* err) -> CUresult {
  if (!err) {
    return CUDA_SUCCESS;
  }

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

CUresult cuModuleLoad(CUmodule* module, const char* path) {
  if (!module) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!path) {
    return CUDA_ERROR_FILE_NOT_FOUND;
  }

  const auto u8_path = NS::String::string(path, NS::UTF8StringEncoding);

  auto err = static_cast<NS::Error*>(nullptr);
  auto& device = CUdevice_st::global();
  auto library = device.newLibrary(u8_path, &err);
  if (err) {
    return toCudaError(err);
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
