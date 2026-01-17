#include "cuda.h"
#include "metal.h"

static void setComputeParams(MTL::ComputeCommandEncoder& encoder, const CUParam params[]) {
  static constexpr auto MAX_PARAMS = 64;
  static auto& device = CUdevice_st::global();

  for (auto i = 0; i < MAX_PARAMS; ++i) {
    auto& p = params[i];
    switch (p._type) {
      case CUParam::None: {
        break;
      }

      case CUParam::Bytes: {
        encoder.setBytes(p._data, p._size, i);
        break;
      }
      case CUParam::Buffer:
        if (auto buff = device.findBuffer(p._data)) {
          encoder.setBuffer(buff.buffer, buff.offset, i);
        }
        break;
      case CUParam::Texture: {
        const auto texture = static_cast<const MTL::Texture*>(p._data);
        encoder.setTexture(texture, i);
        break;
      }
      case CUParam::Sampler: {
        const auto sampler = static_cast<const MTL::SamplerState*>(p._data);
        encoder.setSamplerState(sampler, i);
        break;
      }
    }
  }
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

  auto& device = CUdevice_st::global();
  if (stream == nullptr) {
    stream = static_cast<CUstream_st*>(device.defaultStream());
  }

  // Get device and create compute pipeline state
  NS::Error* error = nullptr;
  auto pipeline_state = AutoRelease{device.newComputePipelineState(f, &error)};
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
  const auto threadsPerGrid = MTL::Size{gridDimX * blockDimX, gridDimY * blockDimY, gridDimZ * blockDimZ};
  const auto threadsPerThreadGroup = MTL::Size{blockDimX, blockDimY, blockDimZ};
  compute_encoder->dispatchThreads(threadsPerGrid, threadsPerThreadGroup);
  compute_encoder->endEncoding();

  // Commit and wait for completion
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  return CUDA_SUCCESS;
}
