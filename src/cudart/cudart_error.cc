#include "cuda_runtime_api.h"

const char* cudaGetErrorName(cudaError_t error) {
#define X(name) \
  case name: return #name

  switch(error)  {
  X(cudaSuccess);

  // Invalid input
  X(cudaErrorInvalidValue);

  // Memory allocation errors
  X(cudaErrorMemoryAllocation);

  // Initialization errors
  X(cudaErrorInitializationError);
  X(cudaErrorCudartUnloading);

  // Profiler errors
  X(cudaErrorProfilerDisabled);
  X(cudaErrorProfilerNotInitialized);
  X(cudaErrorProfilerAlreadyStarted);
  X(cudaErrorProfilerAlreadyStopped);

  // Configuration errors
  X(cudaErrorInvalidConfiguration);

  // Pitch and symbol errors
  X(cudaErrorInvalidPitchValue);
  X(cudaErrorInvalidSymbol);

  // Pointer errors
  X(cudaErrorInvalidHostPointer);
  X(cudaErrorInvalidDevicePointer);

  // Texture errors
  X(cudaErrorInvalidTexture);
  X(cudaErrorInvalidTextureBinding);
  X(cudaErrorInvalidChannelDescriptor);
  X(cudaErrorInvalidMemcpyDirection);

  // Deprecated errors
  X(cudaErrorAddressOfConstant);
  X(cudaErrorTextureFetchFailed);
  X(cudaErrorTextureNotBound);
  X(cudaErrorSynchronizationError);
  X(cudaErrorInvalidFilterSetting);
  X(cudaErrorInvalidNormSetting);
  X(cudaErrorMixedDeviceExecution);
  X(cudaErrorNotYetImplemented);
  X(cudaErrorMemoryValueTooLarge);

  // Driver errors
  X(cudaErrorStubLibrary);
  X(cudaErrorInsufficientDriver);
  X(cudaErrorCallRequiresNewerDriver);

  // Surface errors
  X(cudaErrorInvalidSurface);

  // Duplicate name errors
  X(cudaErrorDuplicateVariableName);
  X(cudaErrorDuplicateTextureName);
  X(cudaErrorDuplicateSurfaceName);

  // Device availability errors
  X(cudaErrorDevicesUnavailable);
  X(cudaErrorIncompatibleDriverContext);

  // Launch configuration errors
  X(cudaErrorMissingConfiguration);
  X(cudaErrorPriorLaunchFailure);

  // Runtime launch errors
  X(cudaErrorLaunchMaxDepthExceeded);
  X(cudaErrorLaunchFileScopedTex);
  X(cudaErrorLaunchFileScopedSurf);
  X(cudaErrorSyncDepthExceeded);
  X(cudaErrorLaunchPendingCountExceeded);

  // Device function errors
  X(cudaErrorInvalidDeviceFunction);

  // Device errors
  X(cudaErrorNoDevice);
  X(cudaErrorInvalidDevice);
  X(cudaErrorDeviceNotLicensed);
  X(cudaErrorSoftwareValidityNotEstablished);

  // Startup errors
  X(cudaErrorStartupFailure);

  // Kernel image errors
  X(cudaErrorInvalidKernelImage);
  X(cudaErrorDeviceUninitialized);

  // Memory mapping errors
  X(cudaErrorMapBufferObjectFailed);
  X(cudaErrorUnmapBufferObjectFailed);
  X(cudaErrorArrayIsMapped);
  X(cudaErrorAlreadyMapped);
  X(cudaErrorNoKernelImageForDevice);
  X(cudaErrorAlreadyAcquired);
  X(cudaErrorNotMapped);
  X(cudaErrorNotMappedAsArray);
  X(cudaErrorNotMappedAsPointer);

  // ECC and hardware errors
  X(cudaErrorECCUncorrectable);
  X(cudaErrorUnsupportedLimit);
  X(cudaErrorDeviceAlreadyInUse);
  X(cudaErrorPeerAccessUnsupported);

  // PTX/JIT compilation errors
  X(cudaErrorInvalidPtx);
  X(cudaErrorInvalidGraphicsContext);
  X(cudaErrorNvlinkUncorrectable);
  X(cudaErrorJitCompilerNotFound);
  X(cudaErrorUnsupportedPtxVersion);
  X(cudaErrorJitCompilationDisabled);
  X(cudaErrorUnsupportedExecAffinity);
  X(cudaErrorUnsupportedDevSideSync);
  X(cudaErrorContained);

  // Source/Compilation errors
  X(cudaErrorInvalidSource);
  X(cudaErrorFileNotFound);
  X(cudaErrorSharedObjectSymbolNotFound);
  X(cudaErrorSharedObjectInitFailed);
  X(cudaErrorOperatingSystem);

  // Handle errors
  X(cudaErrorInvalidResourceHandle);
  X(cudaErrorIllegalState);
  X(cudaErrorLossyQuery);

  // Symbol errors
  X(cudaErrorSymbolNotFound);

  // Async errors
  X(cudaErrorNotReady);

  // Launch errors
  X(cudaErrorIllegalAddress);
  X(cudaErrorLaunchOutOfResources);
  X(cudaErrorLaunchTimeout);
  X(cudaErrorLaunchIncompatibleTexturing);

  // Peer access errors
  X(cudaErrorPeerAccessAlreadyEnabled);
  X(cudaErrorPeerAccessNotEnabled);

  // Context errors
  X(cudaErrorSetOnActiveProcess);
  X(cudaErrorContextIsDestroyed);
  X(cudaErrorAssert);

  // Resource errors
  X(cudaErrorTooManyPeers);
  X(cudaErrorHostMemoryAlreadyRegistered);
  X(cudaErrorHostMemoryNotRegistered);

  // Hardware errors
  X(cudaErrorHardwareStackError);
  X(cudaErrorIllegalInstruction);
  X(cudaErrorMisalignedAddress);
  X(cudaErrorInvalidAddressSpace);
  X(cudaErrorInvalidPc);
  X(cudaErrorLaunchFailure);
  X(cudaErrorCooperativeLaunchTooLarge);
  X(cudaErrorTensorMemoryLeak);

  // System errors
  X(cudaErrorNotPermitted);
  X(cudaErrorNotSupported);
  X(cudaErrorSystemNotReady);
  X(cudaErrorSystemDriverMismatch);
  X(cudaErrorCompatNotSupportedOnDevice);

  // MPS errors
  X(cudaErrorMpsConnectionFailed);
  X(cudaErrorMpsRpcFailure);
  X(cudaErrorMpsServerNotReady);
  X(cudaErrorMpsMaxClientsReached);
  X(cudaErrorMpsMaxConnectionsReached);
  X(cudaErrorMpsClientTerminated);

  // CDP errors
  X(cudaErrorCdpNotSupported);
  X(cudaErrorCdpVersionMismatch);

  // Stream capture errors
  X(cudaErrorStreamCaptureUnsupported);
  X(cudaErrorStreamCaptureInvalidated);
  X(cudaErrorStreamCaptureMerge);
  X(cudaErrorStreamCaptureUnmatched);
  X(cudaErrorStreamCaptureUnjoined);
  X(cudaErrorStreamCaptureIsolation);
  X(cudaErrorStreamCaptureImplicit);
  X(cudaErrorCapturedEvent);
  X(cudaErrorStreamCaptureWrongThread);

  // Timeout and graph errors
  X(cudaErrorTimeout);
  X(cudaErrorGraphExecUpdateFailure);

  // External errors
  X(cudaErrorExternalDevice);

  // Cluster and resource errors
  X(cudaErrorInvalidClusterSize);
  X(cudaErrorFunctionNotLoaded);
  X(cudaErrorInvalidResourceType);
  X(cudaErrorInvalidResourceConfiguration);

  // Stream detached
  X(cudaErrorStreamDetached);

  // Unknown error
  X(cudaErrorUnknown);

  // API failure base
  X(cudaErrorApiFailureBase);
}

#undef X
}
