#pragma once

#include <metal-cpp/Metal/Metal.hpp>

#include "ns_util.h"

class CUstream_st : public MTL::CommandQueue {};
class CUmod_st : public MTL::Library {};
class CUfunc_st : public MTL::Function {};
class CUarray_st : public MTL::Texture {};
class cudaTextureObject_st : public MTL::SamplerState {};

class MetalCtx {
 public:
  explicit MetalCtx(MTL::Device& device);
  ~MetalCtx();
  MetalCtx(const MetalCtx&) = delete;
  MetalCtx& operator=(const MetalCtx&) = delete;

  static auto global() -> MetalCtx&;

  // library
  auto loadLibrary(NS::String* path, NS::Error** err) -> MTL::Library*;
  auto compileLibrary(NS::String* source, NS::Error** err) -> MTL::Library*;

  // command queue
  auto newCommandQueue() -> MTL::CommandQueue*;
  void delCommandQueue(MTL::CommandQueue* queue);
  auto defaultCommandQueue() -> MTL::CommandQueue*;
  void Synchronize();

  // buffer
  auto newBuffer(NS::UInteger length, MTL::ResourceOptions options = MTL::ResourceStorageModeShared) -> MTL::Buffer*;
  void delBuffer(MTL::Buffer* buffer);
  auto findBuffer(const void* ptr) -> MTL::Buffer*;

  // texture
  auto newTexture(const MTL::TextureDescriptor* desc) -> MTL::Texture*;
  void delTexture(MTL::Texture* texture);

  // sampler
  auto newSamplerState(const MTL::SamplerDescriptor* desc, MTL::Texture* tex) -> MTL::SamplerState*;
  void delSamplerState(MTL::SamplerState* sampler);

  // Deref[Device]
  auto operator->() -> MTL::Device* {
    return &_device;
  }

 private:
  MTL::Device& _device;
  MTL::CommandQueue* _command_queue{nullptr};
  std::vector<MTL::Buffer*> _buffers;
};
