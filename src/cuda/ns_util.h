#pragma once

#include <metal-cpp/Metal/Metal.hpp>

template <class T>
struct AutoRelease {
  T* _ptr = nullptr;

 public:
  AutoRelease(T* ptr = nullptr) : _ptr{ptr} {}

  ~AutoRelease() {
    if (_ptr) {
      _ptr->release();
    }
  }

  AutoRelease(AutoRelease&& other) noexcept : _ptr{other._ptr} {
    other._ptr = nullptr;
  }

  AutoRelease& operator=(AutoRelease&& other) noexcept {
    if (this != &other) {
      std::swap(_ptr, other._ptr);
    }
    return *this;
  }

  operator bool() const noexcept {
    return _ptr != nullptr;
  }

  operator T*() const noexcept {
    return _ptr;
  }

  auto operator*() noexcept -> T& {
    return *_ptr;
  }

  auto operator->() noexcept -> T* {
    return _ptr;
  }
};

struct File {
  static auto readToString(NS::String* path) -> NS::String*;
};
