#pragma once

#include <metal-cpp/Metal/Metal.hpp>

struct BufferRange {
  MTL::Buffer* buffer;
  NS::UInteger offset;

  operator bool() const noexcept {
    return buffer != nullptr;
  }
};

class CUdevice_st : public MTL::Device {
 public:
  static auto global() -> CUdevice_st&;

  // buffer
  auto newBuffer(NS::UInteger length, MTL::ResourceOptions options) -> MTL::Buffer*;
  void delBuffer(MTL::Buffer* buffer);

  // test if ptr is in a buffer managed by this device
  auto findBuffer(const void* ptr) -> BufferRange;

  // texture
  auto newTexture(const MTL::TextureDescriptor* desc) -> MTL::Texture*;
  void delTexture(MTL::Texture* texture);

  // sampler
  auto newSamplerState(const MTL::SamplerDescriptor* desc, MTL::Texture* tex) -> MTL::SamplerState*;
  void delSamplerState(MTL::SamplerState* sampler);
  auto getBoundTexture(const MTL::SamplerState* sampler) const noexcept -> MTL::Texture*;
};

class CUstream_st : public MTL::CommandQueue {
 public:
  static auto global() -> CUstream_st&;
};

struct CUmod_st : MTL::Library {};

struct CUfunc_st : MTL::Function {};

struct cudaArray : MTL::Texture {};

struct cudaTextureObject_st : MTL::SamplerState {};
