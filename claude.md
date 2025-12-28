# MUDA
impl CUDA on MacOS, using metal

## metal-cpp
- source code in `metal-cpp/`
- metal-cpp is a C++ wrapper of Metal API provided by Apple, don't modify it.
- don't look into it, since is just a metal-wrapper.

## cudart
- source code in `src/cudart/`
- cudart impl CUDA runtime API with metal-cpp
- only a support functions listed in include/cudart/*.h
- don't modify cuda_runtime_api.h, since is all functions should be supported.

## cufft
- source code in `src/cufft/`
- cufft impl CUDA FFT library with apple's builtin vDSP
- only a support functions listed in include/cufft/*.h
- only support 1D DFT with (c2c, c2r, r2c)
