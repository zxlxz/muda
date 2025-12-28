#include <mutex>
#include <vector>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "cuda/metal.h"

template <class T>
class DeviceCache {
  std::mutex _mutex;
  std::vector<T> _buffers;

 public:
  static auto instance() -> DeviceCache& {
    static auto res = DeviceCache{};
    return res;
  }

  void addValue(const T& val) {
    auto lock = std::lock_guard{_mutex};
    _buffers.push_back(val);
  }

  void removeIf(auto&& f) {
    auto lock = std::lock_guard{_mutex};

    auto itr = std::find_if(_buffers.begin(), _buffers.end(), f);
    if (itr != _buffers.end()) {
      _buffers.erase(itr);
    }
  }

  auto findIf(auto&& f) -> const T* {
    auto lock = std::lock_guard{_mutex};
    for (auto& x : _buffers) {
      if (f(x)) {
        return &x;
      }
    }
    return nullptr;
  }
};

struct BufferInfo {
  MTL::Buffer* buffer;
  void* contents;
  NS::UInteger length;

 public:
  // range [begin, end)
  auto contains(const void* p) const -> bool {
    const auto ptr = static_cast<const char*>(p);
    const auto begin = static_cast<const char*>(contents);
    const auto end = begin + length;
    return (ptr >= begin) && (ptr < end);
  }
};

struct SamplerInfo {
  MTL::SamplerState* sampler;
  MTL::Texture* texture;
};

auto CUdevice_st::newBuffer(NS::UInteger length, MTL::ResourceOptions options) -> MTL::Buffer* {
  if (length == 0) {
    return nullptr;
  }

  auto buffer = MTL::Device::newBuffer(length, options);
  if (!buffer) {
    return nullptr;
  }

  const auto info = BufferInfo{
      .buffer = buffer,
      .contents = buffer->contents(),
      .length = length,
  };
  auto& cache = DeviceCache<BufferInfo>::instance();
  cache.addValue(info);

  return buffer;
}

void CUdevice_st::delBuffer(MTL::Buffer* buffer) {
  buffer->release();

  auto& cache = DeviceCache<BufferInfo>::instance();
  cache.removeIf([buffer](const BufferInfo& info) { return info.buffer == buffer; });
}

auto CUdevice_st::findBuffer(const void* ptr) -> BufferRange {
  auto& cache = DeviceCache<BufferInfo>::instance();

  auto info = cache.findIf([ptr](const BufferInfo& info) { return info.contains(ptr); });
  if (!info) {
    return BufferRange{nullptr, 0};
  }
  const auto curr_ptr = static_cast<const char*>(ptr);
  const auto base_ptr = static_cast<const char*>(info->contents);
  const auto offset = static_cast<NS::UInteger>(curr_ptr - base_ptr);
  return BufferRange{info->buffer, offset};
}

auto CUdevice_st::newTexture(const MTL::TextureDescriptor* desc) -> MTL::Texture* {
  auto texture = MTL::Device::newTexture(desc);

  auto& cache = DeviceCache<MTL::Texture*>::instance();
  cache.addValue(texture);

  return texture;
}

void CUdevice_st::delTexture(MTL::Texture* texture) {
  if (!texture) {
    return;
  }

  auto& cache = DeviceCache<MTL::Texture*>::instance();
  cache.removeIf([texture](MTL::Texture* t) { return t == texture; });
  texture->release();
}

auto CUdevice_st::newSamplerState(const MTL::SamplerDescriptor* desc, MTL::Texture* tex) -> MTL::SamplerState* {
  auto sampler = MTL::Device::newSamplerState(desc);

  auto& cache = DeviceCache<SamplerInfo>::instance();
  cache.addValue({sampler, tex});

  return sampler;
}

void CUdevice_st::delSamplerState(MTL::SamplerState* sampler) {
  if (!sampler) {
    return;
  }

  auto& cache = DeviceCache<SamplerInfo>::instance();
  cache.removeIf([sampler](const SamplerInfo& info) { return info.sampler == sampler; });
  sampler->release();
}

auto CUdevice_st::getBoundTexture(const MTL::SamplerState* sampler) const noexcept -> MTL::Texture* {
  if (!sampler) {
    return nullptr;
  }

  auto& cache = DeviceCache<SamplerInfo>::instance();
  auto info = cache.findIf([sampler](const SamplerInfo& info) { return info.sampler == sampler; });
  if (info) {
    return info->texture;
  }
  return nullptr;
}

auto CUdevice_st::global() -> CUdevice_st& {
  static auto g_device = MTL::CreateSystemDefaultDevice();
  return static_cast<CUdevice_st&>(*g_device);
}

auto CUstream_st::global() -> CUstream_st& {
  static auto g_stream = CUdevice_st::global().newCommandQueue();
  return static_cast<CUstream_st&>(*g_stream);
}
