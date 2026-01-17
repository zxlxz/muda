#include <mutex>
#include <vector>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "cuda/metal.h"

template <class T>
struct DeviceCache {
  mutable std::mutex _mutex;
  std::vector<T> _buffers;

 public:
  static auto instance() -> DeviceCache& {
    static auto res = DeviceCache{};
    return res;
  }

  void addValue(T val) {
    auto lock = std::lock_guard{_mutex};
    _buffers.push_back(std::move(val));
  }

  void removeIf(auto&& f) {
    auto lock = std::lock_guard{_mutex};

    auto itr = std::find_if(_buffers.begin(), _buffers.end(), f);
    if (itr != _buffers.end()) {
      _buffers.erase(itr);
    }
  }

  auto findIf(auto&& f) const -> const T* {
    auto lock = std::lock_guard{_mutex};
    for (auto& x : _buffers) {
      if (f(x)) {
        return &x;
      }
    }
    return nullptr;
  }
};

struct DeviceInfo {
  CUdevice_st* device;
  AutoRelease<CUstream_st> stream;
};

struct CommandQueueInfo {
  MTL::CommandQueue* queue;
};

struct BufferInfo {
  MTL::Buffer* buffer;
  void* contents;
  NS::UInteger length;

 public:
  // range [begin, end)
  bool contains(const void* p) const {
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

// command queue
auto CUdevice_st::newCommandQueue() -> MTL::CommandQueue* {
  auto command_queue = MTL::Device::newCommandQueue();

  auto& cache = DeviceCache<CommandQueueInfo>::instance();
  cache.addValue({command_queue});

  return command_queue;
}

void CUdevice_st::delCommandQueue(MTL::CommandQueue* queue) {
  if (!queue) {
    return;
  }

  auto& cache = DeviceCache<CommandQueueInfo>::instance();
  cache.removeIf([queue](const CommandQueueInfo& info) { return info.queue == queue; });
  queue->release();
}

auto CUdevice_st::defaultStream() -> CUstream_st* {
  auto& cache = DeviceCache<DeviceInfo>::instance();

  auto findResult = cache.findIf([this](const DeviceInfo& info) { return info.device == this; });
  if (findResult) {
    return findResult->stream;
  }

  auto stream = static_cast<CUstream_st*>(MTL::Device::newCommandQueue());
  cache.addValue(DeviceInfo{this, AutoRelease{stream}});
  return stream;
}

void CUdevice_st::Synchronize() {
  auto command_queue = defaultStream();

  // use a empty command buffer to synchronize
  auto command_buffer = AutoRelease{command_queue->commandBuffer()};
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

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
  static auto g_device = AutoRelease<MTL::Device>{MTL::CreateSystemDefaultDevice()};
  return static_cast<CUdevice_st&>(*g_device);
}
