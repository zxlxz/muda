#pragma once

#include <stdint.h>
#include <stdlib.h>

#pragma region error handling
enum cudaError {
  cudaSuccess = 0,

  // Invalid input
  cudaErrorInvalidValue = 1,

  // Memory allocation errors
  cudaErrorMemoryAllocation = 2,

  // Initialization errors
  cudaErrorInitializationError = 3,
  cudaErrorCudartUnloading = 4,

  // Profiler errors
  cudaErrorProfilerDisabled = 5,
  cudaErrorProfilerNotInitialized = 6,
  cudaErrorProfilerAlreadyStarted = 7,
  cudaErrorProfilerAlreadyStopped = 8,

  // Configuration errors
  cudaErrorInvalidConfiguration = 9,

  // Pitch and symbol errors
  cudaErrorInvalidPitchValue = 12,
  cudaErrorInvalidSymbol = 13,

  // Pointer errors
  cudaErrorInvalidHostPointer = 16,
  cudaErrorInvalidDevicePointer = 17,

  // Texture errors
  cudaErrorInvalidTexture = 18,
  cudaErrorInvalidTextureBinding = 19,
  cudaErrorInvalidChannelDescriptor = 20,
  cudaErrorInvalidMemcpyDirection = 21,

  // Deprecated errors
  cudaErrorAddressOfConstant = 22,
  cudaErrorTextureFetchFailed = 23,
  cudaErrorTextureNotBound = 24,
  cudaErrorSynchronizationError = 25,
  cudaErrorInvalidFilterSetting = 26,
  cudaErrorInvalidNormSetting = 27,
  cudaErrorMixedDeviceExecution = 28,
  cudaErrorNotYetImplemented = 31,
  cudaErrorMemoryValueTooLarge = 32,

  // Driver errors
  cudaErrorStubLibrary = 34,
  cudaErrorInsufficientDriver = 35,
  cudaErrorCallRequiresNewerDriver = 36,

  // Surface errors
  cudaErrorInvalidSurface = 37,

  // Duplicate name errors
  cudaErrorDuplicateVariableName = 43,
  cudaErrorDuplicateTextureName = 44,
  cudaErrorDuplicateSurfaceName = 45,

  // Device availability errors
  cudaErrorDevicesUnavailable = 46,
  cudaErrorIncompatibleDriverContext = 49,

  // Launch configuration errors
  cudaErrorMissingConfiguration = 52,
  cudaErrorPriorLaunchFailure = 53,

  // Runtime launch errors
  cudaErrorLaunchMaxDepthExceeded = 65,
  cudaErrorLaunchFileScopedTex = 66,
  cudaErrorLaunchFileScopedSurf = 67,
  cudaErrorSyncDepthExceeded = 68,
  cudaErrorLaunchPendingCountExceeded = 69,

  // Device function errors
  cudaErrorInvalidDeviceFunction = 98,

  // Device errors
  cudaErrorNoDevice = 100,
  cudaErrorInvalidDevice = 101,
  cudaErrorDeviceNotLicensed = 102,
  cudaErrorSoftwareValidityNotEstablished = 103,

  // Startup errors
  cudaErrorStartupFailure = 127,

  // Kernel image errors
  cudaErrorInvalidKernelImage = 200,
  cudaErrorDeviceUninitialized = 201,

  // Memory mapping errors
  cudaErrorMapBufferObjectFailed = 205,
  cudaErrorUnmapBufferObjectFailed = 206,
  cudaErrorArrayIsMapped = 207,
  cudaErrorAlreadyMapped = 208,
  cudaErrorNoKernelImageForDevice = 209,
  cudaErrorAlreadyAcquired = 210,
  cudaErrorNotMapped = 211,
  cudaErrorNotMappedAsArray = 212,
  cudaErrorNotMappedAsPointer = 213,

  // ECC and hardware errors
  cudaErrorECCUncorrectable = 214,
  cudaErrorUnsupportedLimit = 215,
  cudaErrorDeviceAlreadyInUse = 216,
  cudaErrorPeerAccessUnsupported = 217,

  // PTX/JIT compilation errors
  cudaErrorInvalidPtx = 218,
  cudaErrorInvalidGraphicsContext = 219,
  cudaErrorNvlinkUncorrectable = 220,
  cudaErrorJitCompilerNotFound = 221,
  cudaErrorUnsupportedPtxVersion = 222,
  cudaErrorJitCompilationDisabled = 223,
  cudaErrorUnsupportedExecAffinity = 224,
  cudaErrorUnsupportedDevSideSync = 225,
  cudaErrorContained = 226,

  // Source/Compilation errors
  cudaErrorInvalidSource = 300,
  cudaErrorFileNotFound = 301,
  cudaErrorSharedObjectSymbolNotFound = 302,
  cudaErrorSharedObjectInitFailed = 303,
  cudaErrorOperatingSystem = 304,

  // Handle errors
  cudaErrorInvalidResourceHandle = 400,
  cudaErrorIllegalState = 401,
  cudaErrorLossyQuery = 402,

  // Symbol errors
  cudaErrorSymbolNotFound = 500,

  // Async errors
  cudaErrorNotReady = 600,

  // Launch errors
  cudaErrorIllegalAddress = 700,
  cudaErrorLaunchOutOfResources = 701,
  cudaErrorLaunchTimeout = 702,
  cudaErrorLaunchIncompatibleTexturing = 703,

  // Peer access errors
  cudaErrorPeerAccessAlreadyEnabled = 704,
  cudaErrorPeerAccessNotEnabled = 705,

  // Context errors
  cudaErrorSetOnActiveProcess = 708,
  cudaErrorContextIsDestroyed = 709,
  cudaErrorAssert = 710,

  // Resource errors
  cudaErrorTooManyPeers = 711,
  cudaErrorHostMemoryAlreadyRegistered = 712,
  cudaErrorHostMemoryNotRegistered = 713,

  // Hardware errors
  cudaErrorHardwareStackError = 714,
  cudaErrorIllegalInstruction = 715,
  cudaErrorMisalignedAddress = 716,
  cudaErrorInvalidAddressSpace = 717,
  cudaErrorInvalidPc = 718,
  cudaErrorLaunchFailure = 719,
  cudaErrorCooperativeLaunchTooLarge = 720,
  cudaErrorTensorMemoryLeak = 721,

  // System errors
  cudaErrorNotPermitted = 800,
  cudaErrorNotSupported = 801,
  cudaErrorSystemNotReady = 802,
  cudaErrorSystemDriverMismatch = 803,
  cudaErrorCompatNotSupportedOnDevice = 804,

  // MPS errors
  cudaErrorMpsConnectionFailed = 805,
  cudaErrorMpsRpcFailure = 806,
  cudaErrorMpsServerNotReady = 807,
  cudaErrorMpsMaxClientsReached = 808,
  cudaErrorMpsMaxConnectionsReached = 809,
  cudaErrorMpsClientTerminated = 810,

  // CDP errors
  cudaErrorCdpNotSupported = 811,
  cudaErrorCdpVersionMismatch = 812,

  // Stream capture errors
  cudaErrorStreamCaptureUnsupported = 900,
  cudaErrorStreamCaptureInvalidated = 901,
  cudaErrorStreamCaptureMerge = 902,
  cudaErrorStreamCaptureUnmatched = 903,
  cudaErrorStreamCaptureUnjoined = 904,
  cudaErrorStreamCaptureIsolation = 905,
  cudaErrorStreamCaptureImplicit = 906,
  cudaErrorCapturedEvent = 907,
  cudaErrorStreamCaptureWrongThread = 908,

  // Timeout and graph errors
  cudaErrorTimeout = 909,
  cudaErrorGraphExecUpdateFailure = 910,

  // External errors
  cudaErrorExternalDevice = 911,

  // Cluster and resource errors
  cudaErrorInvalidClusterSize = 912,
  cudaErrorFunctionNotLoaded = 913,
  cudaErrorInvalidResourceType = 914,
  cudaErrorInvalidResourceConfiguration = 915,

  // Stream detached
  cudaErrorStreamDetached = 917,

  // Unknown error
  cudaErrorUnknown = 999,

  // API failure base
  cudaErrorApiFailureBase = 10000,
};

using cudaError_t = enum cudaError;

const char* cudaGetErrorName(cudaError_t error);
#pragma endregion

#pragma region device
struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem;
  int multiProcessorCount;
};

cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
cudaError_t cudaDeviceSynchronize();
#pragma endregion

#pragma region stream management
struct CUstream_st;
using cudaStream_t = struct CUstream_st*;

cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
#pragma endregion

#pragma region memory management
enum cudaMemLocationType {
  cudaMemLocationTypeInvalid = 0,
  cudaMemLocationTypeDevice = 1,
  cudaMemLocationTypeHost = 2,
};

enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4,
};

struct cudaMemLocation {
  cudaMemLocationType type;
  int id;
};

cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMallocManaged(void** devPtr, size_t size);

cudaError_t cudaMemPrefetchAsync(
    const void* devPtr, size_t count, cudaMemLocation location, unsigned int flags, cudaStream_t stream);

cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream);

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);

#pragma endregion

#pragma region array management
struct cudaArray;

using cudaArray_t = struct cudaArray*;

struct cudaExtent {
  size_t width;
  size_t height;
  size_t depth;
};

struct cudaPitchedPtr {
  void* ptr;
  size_t pitch;
  size_t xsize;
  size_t ysize;
};

struct cudaMemcpy3DParms {
  cudaPitchedPtr srcPtr;  // host or device ptr
  cudaArray* dstArray;    // device array
  cudaExtent extent;      // width, height, depth
  cudaMemcpyKind kind;    // direction
};

enum cudaChannelFormatKind {
  cudaChannelFormatKindSigned = 0,
  cudaChannelFormatKindUnsigned = 1,
  cudaChannelFormatKindFloat = 2,
  cudaChannelFormatKindNone = 3,
};

struct cudaChannelFormatDesc {
  int x;
  int y;
  int z;
  int w;
  cudaChannelFormatKind f;
};

enum cudaArrayFlags {
  cudaArrayDefault = 0,
  cudaArrayLayered = 1,
  cudaArrayCubemap = 2,
  cudaArraySurfaceLoadStore = 4,
  cudaArrayTextureGather = 8,
};

cudaError_t cudaMalloc3DArray(cudaArray_t* array,
                              const cudaChannelFormatDesc* desc,
                              cudaExtent extent,
                              cudaArrayFlags flags);

cudaError_t cudaFreeArray(cudaArray_t array);

cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, int* flags, cudaArray_t array);

cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms* p);

cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream);

#pragma endregion

#pragma region texture
struct cudaTextureObject_st;

using cudaTextureObject_t = struct cudaTextureObject_st*;

enum cudaResourceType {
  cudaResourceTypeArray = 0,
  cudaResourceTypeLinear = 1,
  cudaResourceTypePitch2D = 2,
};

enum cudaReadMode {
  cudaReadModeElementType = 0,
  cudaReadModeNormalizedFloat = 1,
};

enum cudaTextureAddressMode {
  cudaAddressModeWrap = 0,
  cudaAddressModeClamp = 1,
  cudaAddressModeMirror = 2,
  cudaAddressModeBorder = 3,
};

enum cudaTextureFilterMode {
  cudaFilterModePoint = 0,
  cudaFilterModeLinear = 1,
};

struct cudaResourceDesc {
  cudaResourceType resType;
  union {
    struct {
      struct cudaArray* array;
    } array;
  } res;
};

struct cudaTextureDesc {
  cudaTextureAddressMode addressMode[3];
  cudaTextureFilterMode filterMode;
  cudaReadMode readMode;
  int normalizedCoords;
};

struct cudaResourceViewDesc;

cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject,
                                    const cudaResourceDesc* pResDesc,
                                    const cudaTextureDesc* pTexDesc,
                                    const cudaResourceViewDesc* pResViewDesc);

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject);
#pragma endregion
