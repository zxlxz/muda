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
  const auto N = 1024U;

  // 1. alloc memory
  uint32_t* a = nullptr;
  SAFE_CALL(cudaMalloc((void**)&a, N * sizeof(uint32_t)));

  // 2. memset
  SAFE_CALL(cudaMemset(a, 1, N * sizeof(uint32_t)));

  // 3. test memory
  for (auto i = 0U; i < N; ++i) {
    if (a[i] != 0x01010101) {
      std::println("memset failed at index {}, value={}", i, a[i]);
      return -1;
    }
  }
  // 4. free memory
  SAFE_CALL(cudaFree(a));
}
