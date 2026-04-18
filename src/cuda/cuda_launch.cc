#include "cuda.h"
#include "metal.h"

static void setComputeParams(MTL::ComputeCommandEncoder& encoder, const CUParam_st params[]) {
  static constexpr auto MAX_PARAMS = 64;
  static auto& ctx = MetalCtx::global();

  for (auto i = 0; i < MAX_PARAMS; ++i) {
    auto& p = params[i];
    switch (p._type) {
      case CUParam_st::None: {
        break;
      }

      case CUParam_st::Bytes: {
        encoder.setBytes(p._data, p._size, i);
        break;
      }
      case CUParam_st::Buffer:
        if (auto buff = ctx.findBuffer(p._data)) {
          const auto base = static_cast<const char*>(buff->contents());
          const auto data = static_cast<const char*>(p._data);
          const auto offset = data - base;
          encoder.setBuffer(buff, offset, i);
        }
        break;
      case CUParam_st::Texture: {
        const auto texture = static_cast<const MTL::Texture*>(p._data);
        encoder.setTexture(texture, i);
        break;
      }
      case CUParam_st::Sampler: {
        const auto sampler = static_cast<const MTL::SamplerState*>(p._data);
        encoder.setSamplerState(sampler, i);
        break;
      }
    }
  }
}

CUresult cuLaunchKernelEx(const CUlaunchConfig* conf, CUfunction f, const CUParam_st params[], void** /*extra*/) {
  if (!f || !conf) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto& device = MetalCtx::global();
  auto stream = conf->hStream;
  if (stream == nullptr) {
    stream = static_cast<CUstream_st*>(device.defaultCommandQueue());
  }

  // Get device and create compute pipeline state
  NS::Error* error = nullptr;
  auto pipeline_state = AutoRelease{device->newComputePipelineState(f, &error)};
  if (!pipeline_state) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Create command buffer and compute encoder
  auto command_buffer = AutoRelease{stream->commandBuffer()};
  if (!command_buffer) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Create compute command encoder
  auto compute_encoder = AutoRelease{command_buffer->computeCommandEncoder()};
  if (!compute_encoder) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Set compute pipeline state
  compute_encoder->setComputePipelineState(pipeline_state);
  setComputeParams(*compute_encoder, params);

  // Set threadgroups and threads per threadgroup
  const auto threadsPerGrid =
      MTL::Size{conf->gridDimX * conf->blockDimX, conf->gridDimY * conf->blockDimY, conf->gridDimZ * conf->blockDimZ};
  const auto threadsPerThreadGroup = MTL::Size{conf->blockDimX, conf->blockDimY, conf->blockDimZ};
  compute_encoder->dispatchThreads(threadsPerGrid, threadsPerThreadGroup);
  compute_encoder->endEncoding();

  // Commit and wait for completion
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  return CUDA_SUCCESS;
}
