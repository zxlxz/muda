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

  // 4. H2D memcpy
  auto* h = static_cast<uint32_t*>(malloc(N * sizeof(uint32_t)));
  for (auto i = 0U; i < N; ++i) {
    h[i] = i;
  }
  SAFE_CALL(cudaMemcpy(a, h, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

  // 5. D2D memcpy
  uint32_t* b = nullptr;
  SAFE_CALL(cudaMalloc((void**)&b, N * sizeof(uint32_t)));
  SAFE_CALL(cudaMemcpy(b, a, N * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

  // 6. D2H memcpy
  auto* h2 = static_cast<uint32_t*>(malloc(N * sizeof(uint32_t)));
  SAFE_CALL(cudaMemcpy(h2, b, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  // 7. test memory
  for (auto i = 0U; i < N; ++i) {
    if (h2[i] != i) {
      std::println("memcpy failed at index {}, value={}", i, h2[i]);
      return -1;
    }
  }

  // 8. free memory
  SAFE_CALL(cudaFree(a));
  SAFE_CALL(cudaFree(b));
  free(h);
  free(h2);
}
