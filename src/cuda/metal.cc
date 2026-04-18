#include <mutex>
#include <vector>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "cuda/metal.h"

MetalCtx::MetalCtx(MTL::Device& device) : _device{device} {}

MetalCtx::~MetalCtx() {
  if (_command_queue) {
    _command_queue->release();
  }
  _device.release();
}

auto MetalCtx::global() -> MetalCtx& {
  static auto res = MetalCtx{*MTL::CreateSystemDefaultDevice()};
  return res;
}

auto MetalCtx::loadLibrary(NS::String* path, NS::Error** err) -> MTL::Library* {
  // check if path = *.metal
  const auto _metal = NS::String::string(".metal", NS::UTF8StringEncoding);
  if (path->rangeOfString(_metal, NS::BackwardsSearch).location != NS::NotFound) {
    const auto source = File::readToString(path);
    if (source) {
      return this->compileLibrary(source, err);
    }
  }

  return _device.newLibrary(path, err);
}

auto MetalCtx::compileLibrary(NS::String* source, NS::Error** pErr) -> MTL::Library* {
  auto opts = MTL::CompileOptions::alloc()->init();
  opts->setLanguageVersion(MTL::LanguageVersion::LanguageVersion3_2);
  auto lib = _device.newLibrary(source, opts, pErr);
  opts->release();
  return lib;
}

auto MetalCtx::newCommandQueue() -> MTL::CommandQueue* {
  auto cmd_queue = _device.newCommandQueue();
  return cmd_queue;
}

void MetalCtx::delCommandQueue(MTL::CommandQueue* cmd_queue) {
  if (!cmd_queue) {
    return;
  }
  cmd_queue->release();
}

auto MetalCtx::defaultCommandQueue() -> MTL::CommandQueue* {
  if (_command_queue == nullptr) {
    _command_queue = _device.newCommandQueue();
  }
  return _command_queue;
}

void MetalCtx::Synchronize() {
  auto command_queue = defaultCommandQueue();

  // use a empty command buffer to synchronize
  auto command_buffer = command_queue->commandBuffer();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  command_buffer->release();
}

auto MetalCtx::newBuffer(NS::UInteger length, MTL::ResourceOptions options) -> MTL::Buffer* {
  if (length == 0) {
    return nullptr;
  }

  auto buffer = _device.newBuffer(length, options);
  if (buffer == nullptr) {
    return nullptr;
  }
  _buffers.push_back(buffer);
  return buffer;
}

void MetalCtx::delBuffer(MTL::Buffer* buffer) {
  if (!buffer) {
    return;
  }
  std::erase_if(_buffers, [buffer](auto& b) { return b == buffer; });
  buffer->release();
}

auto MetalCtx::findBuffer(const void* raw_ptr) -> MTL::Buffer* {
  const auto ptr = static_cast<const char*>(raw_ptr);
  auto itr = std::find_if(_buffers.begin(), _buffers.end(), [&](auto& buf) {
    const auto p = static_cast<const char*>(buf->contents());
    return ptr >= p && ptr < p + buf->length();
  });
  if (itr == _buffers.end()) {
    return nullptr;
  }
  return *itr;
}

auto MetalCtx::newTexture(const MTL::TextureDescriptor* desc) -> MTL::Texture* {
  auto texture = _device.newTexture(desc);
  return texture;
}

void MetalCtx::delTexture(MTL::Texture* texture) {
  if (!texture) {
    return;
  }
  texture->release();
}

auto MetalCtx::newSamplerState(const MTL::SamplerDescriptor* desc, MTL::Texture* tex) -> MTL::SamplerState* {
  auto sampler = _device.newSamplerState(desc);
  return sampler;
}

void MetalCtx::delSamplerState(MTL::SamplerState* sampler) {
  if (!sampler) {
    return;
  }
  sampler->release();
}
