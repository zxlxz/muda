#pragma once

#include <cuda_runtime_api.h>

template <class T>
static auto cudaFormatKind() -> cudaChannelFormatKind {
  if constexpr (__is_floating_point(T)) {
    return cudaChannelFormatKindFloat;
  } else if constexpr (__is_unsigned(T)) {
    return cudaChannelFormatKindUnsigned;
  } else if constexpr (__is_signed(T)) {
    return cudaChannelFormatKindSigned;
  } else {
    static_assert(false, "cudaFormatKind: unsupported type");
  }
}

template <class T>
auto cudaCreateChannelDesc() -> cudaChannelFormatDesc {
  static constexpr auto x = static_cast<int>(sizeof(T) * 8);
  static constexpr auto f = cudaFormatKind<T>();
  return cudaChannelFormatDesc{x, 0, 0, 0, f};
}
