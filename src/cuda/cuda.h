#pragma once

#include <stddef.h>

using size_t = decltype(sizeof(0));

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

#pragma region device
using CUdevice = struct CUdevice_st*;
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetCount(int* count);
CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev);
#pragma endregion

#pragma region context
struct CUcontext_st;
using CUcontext = struct CUcontext_st*;
CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy(CUcontext ctx);

CUresult cuCtxGetCurrent(CUcontext* pctx);
CUresult cuCtxSetCurrent(CUcontext ctx);

CUresult cuCtxPushCurrent(CUcontext ctx);
CUresult cuCtxPopCurrent(CUcontext* pctx);

CUresult cuCtxSynchronize();
#pragma endregion

#pragma region stream
using CUstream = struct CUstream_st*;
CUresult cuStreamCreate(CUstream* phStream, unsigned int flags);
CUresult cuStreamDestroy(CUstream hStream);
CUresult cuStreamSynchronize(CUstream hStream);
#pragma endregion

#pragma region memory
enum CUmemorytype {
  CU_MEMORYTYPE_HOST = 0,
  CU_MEMORYTYPE_DEVICE = 1,
  CU_MEMORYTYPE_ARRAY = 2,
  CU_MEMORYTYPE_UNIFIED = 3,
};

enum class CUmemLocationType {
  CU_MEM_LOCATION_TYPE_DEVICE = 1,
  CU_MEM_LOCATION_TYPE_ARRAY = 2,
  CU_MEM_LOCATION_TYPE_UNIFIED = 3,
};
struct CUmemLocation {
  CUmemLocationType type;
  int id;
};

CUresult cuMemGetInfo(size_t* free, size_t* total);

CUresult cuMemAlloc(void** dptr, size_t bytesize);
CUresult cuMemFree(void* dptr);

CUresult cuMemAllocManaged(void** dptr, size_t bytesize, unsigned int flags);
CUresult cuMemPrefetchAsync(void* devPtr, size_t count, CUmemLocation location, unsigned int flags, CUstream hStream);

CUresult cuMemAllocHost(void** pp, size_t bytesize);
CUresult cuMemFreeHost(void* p);

CUresult cuMemcpy(void* dst, const void* src, size_t bytesize);
CUresult cuMemcpyAsync(void* dst, const void* src, size_t bytesize, CUstream hStream);

CUresult cuMemsetD8(void* dst, unsigned char uc, size_t N);
CUresult cuMemsetD8Async(void* dst, unsigned char uc, size_t N, CUstream hStream);

CUresult cuMemsetD16(void* dst, unsigned short us, size_t N);
CUresult cuMemsetD16Async(void* dst, unsigned short us, size_t N, CUstream hStream);

CUresult cuMemsetD32(void* dst, unsigned int ui, size_t N);
CUresult cuMemsetD32Async(void* dst, unsigned int ui, size_t N, CUstream hStream);

#pragma endregion

#pragma region array
using CUarray = struct CUarray_st*;
struct CUDA_MEMCPY3D {
  unsigned int srcXInBytes, srcY, srcZ;
  unsigned int srcLOD;
  CUmemorytype srcMemoryType;
  const void* srcHost;
  const void* srcDevice;
  CUarray srcArray;
  unsigned int srcPitch;   // ignored when src is array
  unsigned int srcHeight;  // ignored when src is array; may be 0 if Depth==1

  unsigned int dstXInBytes, dstY, dstZ;
  unsigned int dstLOD;
  CUmemorytype dstMemoryType;
  void* dstHost;
  void* dstDevice;
  CUarray dstArray;
  unsigned int dstPitch;   // ignored when dst is array
  unsigned int dstHeight;  // ignored when dst is array; may be 0 if Depth==1

  unsigned int WidthInBytes;
  unsigned int Height;
  unsigned int Depth;
};

enum CUarray_format {
  CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
  CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
  CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
  CU_AD_FORMAT_SIGNED_INT8 = 0x08,
  CU_AD_FORMAT_SIGNED_INT16 = 0x09,
  CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
  CU_AD_FORMAT_HALF = 0x10,
  CU_AD_FORMAT_FLOAT = 0x20,
};

enum CUarray_flags {
  CU_ARRAY_DEFAULT = 0x00,
  CU_ARRAY_LAYERED = 0x01,
  CU_ARRAY_CUBEMAP = 0x02,
  CU_ARRAY_SURFACE_LDST = 0x04,
  CU_ARRAY_TEXTURE_GATHER = 0x08,
};

struct CUDA_ARRAY3D_DESCRIPTOR {
  size_t Width;
  size_t Height;
  size_t Depth;
  CUarray_format Format;
  unsigned int NumChannels;
  CUarray_flags Flags;
};

CUresult cuArray3DCreate(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray);
CUresult cuArrayDestroy(CUarray hArray);

CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray);

CUresult cuMemcpy3D(struct CUDA_MEMCPY3D* pCopy);
CUresult cuMemcpy3DAsync(struct CUDA_MEMCPY3D* pCopy, CUstream hStream);
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

struct CUDA_TEXTURE_DESC {
  CUaddress_mode addressMode[3];
  CUfilter_mode filterMode;
  unsigned int flags;
  unsigned int maxAnisotropy;
  float mipmapLevelBias;
  float minMipmapLevelClamp;
  float maxMipmapLevelClamp;
};

struct CUDA_RESOURCE_DESC {
  CUresourcetype resType;
  union {
    struct {
      CUarray array;
    } array;
  } res;
};

struct CUDA_RESOURCE_VIEW_DESC {
  unsigned int viewFormat;
  size_t width;
  size_t height;
  size_t depth;
  size_t firstMipmapLevel;
  size_t lastMipmapLevel;
  size_t firstLayer;
  size_t lastLayer;
};

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
#pragma endregion

#pragma region function
using CUParam = struct CUParam_st;

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

CUresult cuLaunchKernel(CUfunction f,
                        unsigned gridDimX,
                        unsigned gridDimY,
                        unsigned gridDimZ,
                        unsigned blockDimX,
                        unsigned blockDimY,
                        unsigned blockDimZ,
                        unsigned sharedMemBytes,
                        CUstream hStream,
                        const CUParam params[],
                        void** extra);
#pragma endregion
