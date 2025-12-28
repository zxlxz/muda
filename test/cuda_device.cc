#include <cuda_runtime_api.h>

#include <print>

#define SAFE_CALL(f)                                                         \
  ({                                                                         \
    const auto err = f;                                                      \
    if (err != cudaSuccess) {                                                \
      std::println("call ({}) failed, error={}", #f, static_cast<int>(err)); \
      return -1;                                                             \
    }                                                                        \
  })

int main(int argc, const char* argv[]) {
  // device cnt
  auto device_cnt = 0;
  SAFE_CALL(cudaGetDeviceCount(&device_cnt));
  std::println("device count: {}", device_cnt);

  // device prop
  for (auto dev_id = 0; dev_id < device_cnt; ++dev_id) {
    auto prop = cudaDeviceProp{};
    SAFE_CALL(cudaGetDeviceProperties(&prop, dev_id));
    std::println("================= Device {} ================", dev_id);
    std::println("device.name:                {}", prop.name);
    std::println("device.totalGlobalMem:      {}", prop.totalGlobalMem);
    std::println("device.multiProcessorCount: {}", prop.multiProcessorCount);
  }

  return 0;
}
