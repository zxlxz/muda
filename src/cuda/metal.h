#pragma once

#include <metal-cpp/Metal/Metal.hpp>

template <class T>
struct AutoRelease {
  T* _ptr = nullptr;

 public:
  AutoRelease(T* ptr = nullptr) : _ptr{ptr} {}

  ~AutoRelease() {
    if (_ptr) {
      _ptr->release();
    }
  }

  AutoRelease(AutoRelease&& other) noexcept : _ptr{other._ptr} {
    other._ptr = nullptr;
  }

  AutoRelease& operator=(AutoRelease&& other) noexcept {
    if (this != &other) {
      std::swap(_ptr, other._ptr);
    }
    return *this;
  }

  operator bool() const noexcept {
    return _ptr != nullptr;
  }

  operator T*() const noexcept {
    return _ptr;
  }

  auto operator*() noexcept -> T& {
    return *_ptr;
  }

  auto operator->() noexcept -> T* {
    return _ptr;
  }
};

struct BufferRange {
  MTL::Buffer* buffer;
  NS::UInteger offset;

  operator bool() const noexcept {
    return buffer != nullptr;
  }
};

class CUstream_st : public MTL::CommandQueue {};
class CUmod_st : public MTL::Library {};
class CUfunc_st : public MTL::Function {};
class CUarray_st : public MTL::Texture {};
class cudaTextureObject_st : public MTL::SamplerState {};

class CUdevice_st : public MTL::Device {
 public:
  static auto global() -> CUdevice_st&;

  // command queue
  auto newCommandQueue() -> MTL::CommandQueue*;
  void delCommandQueue(MTL::CommandQueue* queue);
  auto defaultStream() -> CUstream_st*;
  void Synchronize();

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
