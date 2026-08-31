#pragma once

#include <stddef.h>
#include <stdlib.h>

enum CUresult {
  CUDA_SUCCESS = 0,

  // Invalid input
  CUDA_ERROR_INVALID_VALUE = 1,

  // Memory allocation errors
  CUDA_ERROR_OUT_OF_MEMORY = 2,

  // Initialization errors
  CUDA_ERROR_NOT_INITIALIZED = 3,
  CUDA_ERROR_DEINITIALIZED = 4,

  // Profiler errors
  CUDA_ERROR_PROFILER_DISABLED = 5,
  CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
  CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
  CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,

  // Driver errors
  CUDA_ERROR_STUB_LIBRARY = 34,
  CUDA_ERROR_CALL_REQUIRES_NEWER_DRIVER = 36,
  CUDA_ERROR_DEVICE_UNAVAILABLE = 46,

  // Device errors
  CUDA_ERROR_NO_DEVICE = 100,
  CUDA_ERROR_INVALID_DEVICE = 101,
  CUDA_ERROR_DEVICE_NOT_LICENSED = 102,

  // Image/Module errors
  CUDA_ERROR_INVALID_IMAGE = 200,
  CUDA_ERROR_INVALID_CONTEXT = 201,
  CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,

  // Memory mapping errors
  CUDA_ERROR_MAP_FAILED = 205,
  CUDA_ERROR_UNMAP_FAILED = 206,
  CUDA_ERROR_ARRAY_IS_MAPPED = 207,
  CUDA_ERROR_ALREADY_MAPPED = 208,
  CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
  CUDA_ERROR_ALREADY_ACQUIRED = 210,
  CUDA_ERROR_NOT_MAPPED = 211,
  CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
  CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,

  // ECC and hardware errors
  CUDA_ERROR_ECC_UNCORRECTABLE = 214,
  CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
  CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
  CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,

  // PTX/JIT compilation errors
  CUDA_ERROR_INVALID_PTX = 218,
  CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
  CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
  CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
  CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
  CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
  CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
  CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225,
  CUDA_ERROR_CONTAINED = 226,

  // Source/Compilation errors
  CUDA_ERROR_INVALID_SOURCE = 300,
  CUDA_ERROR_FILE_NOT_FOUND = 301,
  CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
  CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
  CUDA_ERROR_OPERATING_SYSTEM = 304,

  // Handle errors
  CUDA_ERROR_INVALID_HANDLE = 400,
  CUDA_ERROR_ILLEGAL_STATE = 401,
  CUDA_ERROR_LOSSY_QUERY = 402,

  // Symbol errors
  CUDA_ERROR_NOT_FOUND = 500,

  // Async errors
  CUDA_ERROR_NOT_READY = 600,

  // Launch errors
  CUDA_ERROR_ILLEGAL_ADDRESS = 700,
  CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
  CUDA_ERROR_LAUNCH_TIMEOUT = 702,
  CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,

  // Peer access errors
  CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
  CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,

  // Context errors
  CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
  CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
  CUDA_ERROR_ASSERT = 710,

  // Resource errors
  CUDA_ERROR_TOO_MANY_PEERS = 711,
  CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
  CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,

  // Hardware errors
  CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
  CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
  CUDA_ERROR_MISALIGNED_ADDRESS = 716,
  CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
  CUDA_ERROR_INVALID_PC = 718,
  CUDA_ERROR_LAUNCH_FAILED = 719,
  CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
  CUDA_ERROR_TENSOR_MEMORY_LEAK = 721,

  // System errors
  CUDA_ERROR_NOT_PERMITTED = 800,
  CUDA_ERROR_NOT_SUPPORTED = 801,
  CUDA_ERROR_SYSTEM_NOT_READY = 802,
  CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
  CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,

  // MPS errors
  CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
  CUDA_ERROR_MPS_RPC_FAILURE = 806,
  CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
  CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
  CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
  CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,

  // CDP errors
  CUDA_ERROR_CDP_NOT_SUPPORTED = 811,
  CUDA_ERROR_CDP_VERSION_MISMATCH = 812,

  // Stream capture errors
  CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
  CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
  CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
  CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
  CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
  CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
  CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
  CUDA_ERROR_CAPTURED_EVENT = 907,
  CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,

  // Timeout and graph errors
  CUDA_ERROR_TIMEOUT = 909,
  CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,

  // External errors
  CUDA_ERROR_EXTERNAL_DEVICE = 911,

  // Cluster and resource errors
  CUDA_ERROR_INVALID_CLUSTER_SIZE = 912,
  CUDA_ERROR_FUNCTION_NOT_LOADED = 913,
  CUDA_ERROR_INVALID_RESOURCE_TYPE = 914,
  CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION = 915,
  CUDA_ERROR_KEY_ROTATION = 916,
  CUDA_ERROR_STREAM_DETACHED = 917,

  // Unknown error
  CUDA_ERROR_UNKNOWN = 999,
};

#pragma region context
CUresult cuDriverGetVersion(int* driverVersion);
CUresult cuInit(unsigned int flags);
#pragma endregion

#pragma region error
CUresult cuGetErrorName(CUresult error, const char** pStr);
CUresult cuGetErrorString(CUresult error, const char** pStr);
#pragma endregion

#pragma region device
using CUdevice = int;

enum CUdevice_attribute {
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
  CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
  CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
};

CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetCount(int* count);
CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev);
CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev);
#pragma endregion

#pragma region context
struct CUcontext_st;
using CUcontext = struct CUcontext_st*;
CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy(CUcontext ctx);

CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);
CUresult cuDevicePrimaryCtxRelease(CUdevice dev);

CUresult cuCtxGetCurrent(CUcontext* pctx);
CUresult cuCtxSetCurrent(CUcontext ctx);

CUresult cuCtxPushCurrent(CUcontext ctx);
CUresult cuCtxPopCurrent(CUcontext* pctx);

CUresult cuCtxSynchronize();
CUresult cuCtxGetDevice(CUdevice* device);
#pragma endregion

#pragma region stream
using CUstream = struct CUstream_st*;
CUresult cuStreamCreate(CUstream* phStream, unsigned int flags);
CUresult cuStreamDestroy_v2(CUstream hStream);
CUresult cuStreamSynchronize(CUstream hStream);
#pragma endregion

#pragma region memory
using CUdeviceptr = uintptr_t;

enum CUmemorytype {
  CU_MEMORYTYPE_HOST = 0,
  CU_MEMORYTYPE_DEVICE = 1,
  CU_MEMORYTYPE_ARRAY = 2,
  CU_MEMORYTYPE_UNIFIED = 3,
};

enum CUmemLocationType {
  CU_MEM_LOCATION_TYPE_HOST = 0,
  CU_MEM_LOCATION_TYPE_DEVICE = 1,
  CU_MEM_LOCATION_TYPE_ARRAY = 2,
  CU_MEM_LOCATION_TYPE_UNIFIED = 3,
};

enum CUmemAttach_flags {
  CU_MEM_ATTACH_GLOBAL = 0x1,
  CU_MEM_ATTACH_HOST = 0x2,
  CU_MEM_ATTACH_SINGLE = 0x4,
};

struct CUmemLocation {
  CUmemLocationType type;
  int id;
};

enum CUpointer_attribute {
  CU_POINTER_ATTRIBUTE_CONTEXT,
  CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
  CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
  CU_POINTER_ATTRIBUTE_HOST_POINTER,
};

CUresult cuMemGetInfo(size_t* free, size_t* total);
CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr);

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize);
CUresult cuMemFree_v2(CUdeviceptr dptr);

CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags);
CUresult cuMemPrefetchAsync_v2(CUdeviceptr devPtr,
                               size_t count,
                               CUmemLocation location,
                               unsigned int flags,
                               CUstream hStream);

enum CUmemhostalloc_flags {
  CU_MEMHOSTALLOC_PORTABLE = 0x01,
  CU_MEMHOSTALLOC_DEVICEMAP = 0x02,
  CU_MEMHOSTALLOC_WRITECOMBINED = 0x04,
};

CUresult cuMemFreeHost(void* p);
CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int flags);

CUresult cuMemcpy(CUdeviceptr dst, const CUdeviceptr src, size_t bytesize);
CUresult cuMemcpyAsync(CUdeviceptr dst, const CUdeviceptr src, size_t bytesize, CUstream hStream);

CUresult cuMemcpyHtoD(CUdeviceptr dst, const void* src, size_t bytesize);
CUresult cuMemcpyHtoDAsync(CUdeviceptr dst, const void* src, size_t bytesize, CUstream hStream);

CUresult cuMemcpyDtoH(void* dst, const CUdeviceptr src, size_t bytesize);
CUresult cuMemcpyDtoHAsync(void* dst, const CUdeviceptr src, size_t bytesize, CUstream hStream);

CUresult cuMemcpyDtoD(CUdeviceptr dst, const CUdeviceptr src, size_t bytesize);
CUresult cuMemcpyDtoDAsync(CUdeviceptr dst, const CUdeviceptr src, size_t bytesize, CUstream hStream);

CUresult cuMemsetD8_v2(CUdeviceptr dst, unsigned char uc, size_t N);
CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char uc, size_t N, CUstream hStream);

CUresult cuMemsetD16_v2(CUdeviceptr dst, unsigned short us, size_t N);
CUresult cuMemsetD16Async(CUdeviceptr dst, unsigned short us, size_t N, CUstream hStream);

CUresult cuMemsetD32_v2(CUdeviceptr dst, unsigned int ui, size_t N);
CUresult cuMemsetD32Async(CUdeviceptr dst, unsigned int ui, size_t N, CUstream hStream);

#pragma endregion

#pragma region array
using CUarray = struct CUarray_st*;

struct CUDA_MEMCPY3D {
  size_t srcXInBytes, srcY, srcZ;
  size_t srcLOD;
  CUmemorytype srcMemoryType;
  const void* srcHost;
  CUdeviceptr srcDevice;
  CUarray srcArray;
  size_t srcPitch;   // ignored when src is array
  size_t srcHeight;  // ignored when src is array; may be 0 if Depth==1

  size_t dstXInBytes, dstY, dstZ;
  size_t dstLOD;
  CUmemorytype dstMemoryType;
  void* dstHost;
  CUdeviceptr dstDevice;
  CUarray dstArray;
  size_t dstPitch;   // ignored when dst is array
  size_t dstHeight;  // ignored when dst is array; may be 0 if Depth==1

  size_t WidthInBytes;
  size_t Height;
  size_t Depth;
};
using CUDA_MEMCPY3D_st = CUDA_MEMCPY3D;

enum CUarray_format {
  CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
  CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
  CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
  CU_AD_FORMAT_SIGNED_INT8 = 0x08,
  CU_AD_FORMAT_SIGNED_INT16 = 0x09,
  CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
  CU_AD_FORMAT_HALF = 0x10,
  CU_AD_FORMAT_FLOAT = 0x20,

  CU_AD_FORMAT_MAX = 0xFFFFU,
};

enum CUarray3d_flags {
  CUDA_ARRAY3D_DEFAULT = 0x00,
  CUDA_ARRAY3D_LAYERED = 0x01,
  CUDA_ARRAY3D_CUBEMAP = 0x02,
  CUDA_ARRAY3D_SURFACE_LDST = 0x04,
  CUDA_ARRAY3D_TEXTURE_GATHER = 0x08,
};

typedef struct CUDA_ARRAY3D_DESCRIPTOR_st CUDA_ARRAY3D_DESCRIPTOR;
struct CUDA_ARRAY3D_DESCRIPTOR_st {
  size_t Width;
  size_t Height;
  size_t Depth;
  CUarray_format Format;
  unsigned int NumChannels;
  unsigned int Flags;
};

CUresult cuArray3DCreate_v2(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray);
CUresult cuArrayDestroy(CUarray hArray);

CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray);

CUresult cuMemcpy3D_v2(CUDA_MEMCPY3D* pCopy);
CUresult cuMemcpy3DAsync_v2(CUDA_MEMCPY3D* pCopy, CUstream hStream);
#pragma endregion

#pragma region texture
using CUtexObject = unsigned long long;

enum CUfilter_mode {
  CU_TR_FILTER_MODE_POINT = 0,
  CU_TR_FILTER_MODE_LINEAR = 1,
};

enum CUaddress_mode {
  CU_TR_ADDRESS_MODE_WRAP = 0,
  CU_TR_ADDRESS_MODE_CLAMP = 1,
  CU_TR_ADDRESS_MODE_MIRROR = 2,
  CU_TR_ADDRESS_MODE_BORDER = 3,
};

enum CUresourcetype {
  CU_RESOURCE_TYPE_ARRAY = 0,
  CU_RESOURCE_TYPE_LINEAR = 1,
  CU_RESOURCE_TYPE_PITCH2D = 2,
};

enum CUtrsf_flags {
  CU_TRSF_NORMALIZED_COORDINATES = 0x1,
};

typedef struct CUDA_TEXTURE_DESC_st CUDA_TEXTURE_DESC;
struct CUDA_TEXTURE_DESC_st {
  CUaddress_mode addressMode[3];
  CUfilter_mode filterMode;
  unsigned int flags;
  unsigned int maxAnisotropy;
  float mipmapLevelBias;
  float minMipmapLevelClamp;
  float maxMipmapLevelClamp;
};

typedef struct CUDA_RESOURCE_DESC_st CUDA_RESOURCE_DESC;
struct CUDA_RESOURCE_DESC_st {
  CUresourcetype resType;
  union {
    struct {
      CUarray hArray;
    } array;
  } res;
};

struct CUDA_RESOURCE_VIEW_DESC_st;

typedef struct CUDA_RESOURCE_VIEW_DESC_st CUDA_RESOURCE_VIEW_DESC;

CUresult cuTexObjectCreate(CUtexObject* pTexObject,
                           const CUDA_RESOURCE_DESC* pResDesc,
                           const CUDA_TEXTURE_DESC* pTexDesc,
                           const CUDA_RESOURCE_VIEW_DESC* pResViewDesc);
CUresult cuTexObjectDestroy(CUtexObject texObject);
#pragma endregion

#pragma region model
using CUmodule = struct CUmod_st*;
using CUfunction = struct CUfunc_st*;

CUresult cuModuleLoad(CUmodule* module, const char* path);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);

enum CUjit_option {
  CU_JIT_OPTION_END = 0,
};

enum CUlibraryOption {
  CU_LIBRARY_OPTION_END = 0,
};

using CUlibrary = struct CUlib_st*;
using CUkernel = struct CUkern_st*;

CUresult cuLibraryLoadFromFile(CUlibrary* library,
                               const char* fileName,
                               CUjit_option* jitOptions,
                               void** jitOptionsValues,
                               unsigned int numJitOptions,
                               CUlibraryOption* libraryOptions,
                               void** libraryOptionValues,
                               unsigned int numLibraryOptions);
CUresult cuLibraryUnload(CUlibrary library);
CUresult cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name);
CUresult cuKernelGetFunction(CUfunction* pFunction, CUkernel kernel);
#pragma endregion

#pragma region function
struct CUParam_st {
  enum Type { None, Bytes, Buffer, Texture, Sampler };
  Type _type = Type::None;
  unsigned _size = 0;
  const void* _data = nullptr;

 public:
  CUParam_st() noexcept : _type{Type::None}, _size{0}, _data{nullptr} {}

  template <typename T>
  CUParam_st(const T& val, Type t = Bytes) : _type{t}, _size{sizeof(T)}, _data{&val} {}

  template <class T>
  CUParam_st(const T* ptr, Type t = Buffer) : _type{t}, _size{sizeof(T)}, _data{ptr} {}

  template <class T>
  CUParam_st(T* ptr, Type t = Buffer) : _type{t}, _size{sizeof(T)}, _data{ptr} {}
};

struct CUlaunchAttribute;
struct CUlaunchConfig {
  unsigned gridDimX;
  unsigned gridDimY;
  unsigned gridDimZ;
  unsigned blockDimX;
  unsigned blockDimY;
  unsigned blockDimZ;
  unsigned sharedMemBytes;
  CUstream hStream;
  CUlaunchAttribute* attrs;
  unsigned numAttrs;
};

CUresult cuLaunchKernelEx(const CUlaunchConfig* conf, CUfunction f, void* params[], void** extra);
CUresult cuLaunchKernelEx(const CUlaunchConfig* conf, CUfunction f, const CUParam_st params[], void** extra);
#pragma endregion
