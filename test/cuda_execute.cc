#include <cuda.h>
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
  // active cuda runtime
  SAFE_CALL(cudaFree(nullptr));

  // load model
  const auto path = "./axpy.metallib";
  auto mod = CUmodule{};
  if (auto res = cuModuleLoad(&mod, path); res != 0) {
    std::println("cuModuleLoad failed, error={}", static_cast<int>(res));
    return -1;
  }

  // get function
  auto func = CUfunction{};
  if (auto res = cuModuleGetFunction(&func, mod, "_mk_axpy"); res != 0) {
    std::println("cuModuleGetFunction failed, error={}", static_cast<int>(res));
    return -1;
  }

  // args
  const float alpha = 0.1f;
  const uint32_t n = 1024;
  float* X = nullptr;
  float* Y = nullptr;
  SAFE_CALL(cudaMalloc((void**)&X, n * sizeof(float)));
  SAFE_CALL(cudaMalloc((void**)&Y, n * sizeof(float)));
  for (auto i = 0U; i < n; ++i) {
    X[i] = static_cast<float>(i);
    Y[i] = static_cast<float>(i);
  }

  // exec kernel
  const CUParam args[] = {
      alpha, X, Y, n, {},
  };

  cuLaunchKernel(func,      // func
                 16, 1, 1,  // grid dim
                 64, 1, 1,  // block dim
                 0,         // shared mem
                 nullptr,   // stream
                 args,      // kernel params
                 nullptr    // extra
  );

  // wait
  SAFE_CALL(cudaStreamSynchronize(nullptr));

  // verify result
  for (auto i = 0U; i < 10; ++i) {
    printf("X[%u]=%.2f, Y[%u]=%.2f\n", i, X[i], i, Y[i]);
  }

  // dealloc memory
  SAFE_CALL(cudaFree(X));
  SAFE_CALL(cudaFree(Y));

  return 0;
}
