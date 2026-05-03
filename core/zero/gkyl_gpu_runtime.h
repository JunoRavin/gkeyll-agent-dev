#pragma once

// GPU runtime translation shim.
//
// Provides a single include point that resolves to either the CUDA or
// the HIP runtime header, and aliases the CUDA Runtime API symbols used
// by Gkeyll to their HIP equivalents when building with hipcc.
//
// Build-flag gating (set via CFLAGS by the top-level Makefile):
//   GKYL_HAVE_HIP   -> HIP / ROCm path
//   GKYL_HAVE_CUDA  -> CUDA path
//
// Either implies GKYL_HAVE_GPU.

#if defined(GKYL_HAVE_HIP)

// Header selection rules:
//   - C TUs (host code under Cray cc): include only <hip/hip_runtime_api.h>.
//     This gives the host runtime API (hipMalloc, hipMemcpy, hipStream_t, etc.)
//     without dragging in libstdc++. The C++ template helpers in
//     amd_detail/host_defines.h are gated on __HIP__, which only hipcc sets,
//     so a pure-C TU never sees them.
//   - C++ TUs (.cu files under hipcc): include the full <hip/hip_runtime.h>.
//     Device intrinsics (threadIdx, blockIdx, blockDim, gridDim, atomic*,
//     math intrinsics) live there; <hip/hip_runtime_api.h> alone is not
//     enough to compile a kernel. The full header transitively pulls in C++
//     stdlib bits (<memory>, <bits/unique_ptr.h>) which carry templates;
//     because every _cu.cu wraps its Gkyl C-header includes in an
//     `extern "C" {}` block, the shim is reached with C linkage. Escape it
//     with extern "C++" so the templates and stdlib bits parse correctly.
#if defined(__cplusplus)
extern "C++" {
#include <hip/hip_runtime.h>
}
#else
#include <hip/hip_runtime_api.h>
#endif

// Memory management
#define cudaMalloc                  hipMalloc
#define cudaFree                    hipFree
#define cudaMemcpy                  hipMemcpy
#define cudaMemcpyAsync             hipMemcpyAsync
#define cudaMemset                  hipMemset
// hipHostMalloc takes (ptr, size, flags) whereas cudaMallocHost takes (ptr, size).
// hipHostMallocDefault == 0 matches cudaMallocHost's default behavior, so the
// 2-arg call expands to a 3-arg hipHostMalloc with flags=0.
#define cudaMallocHost(ptr, size)   hipHostMalloc((ptr), (size), 0)
#define cudaFreeHost                hipHostFree

// Streams
#define cudaStream_t                hipStream_t
#define cudaStreamCreate            hipStreamCreate
#define cudaStreamDestroy           hipStreamDestroy
#define cudaStreamSynchronize       hipStreamSynchronize
#define cudaDeviceSynchronize       hipDeviceSynchronize

// Errors
#define cudaSuccess                 hipSuccess
#define cudaError_t                 hipError_t
#define cudaGetLastError            hipGetLastError
#define cudaGetErrorString          hipGetErrorString

// Memcpy direction enum members
#define cudaMemcpyHostToHost        hipMemcpyHostToHost
#define cudaMemcpyHostToDevice      hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost      hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice    hipMemcpyDeviceToDevice

// Device management
#define cudaDeviceReset             hipDeviceReset
#define cudaGetDeviceCount          hipGetDeviceCount
#define cudaSetDevice               hipSetDevice

// Intrinsics that exist under the same name on both platforms (documentation only).
// atomicCAS, __double_as_longlong, __longlong_as_double need no rename.

// Kernel files across core/ker/ gate device-only intrinsics (atomicAdd,
// __double_as_longlong, etc.) on `#ifdef __CUDA_ARCH__`. nvcc defines
// that during the device-compile pass; hipcc instead defines
// `__HIP_DEVICE_COMPILE__`. Without translation, those #ifdefs fall to
// the host branch under HIP and atomic accumulators degrade to plain
// `+=` — racy across threads, silent wrong-answer bug.
//
// Make the existing `#ifdef __CUDA_ARCH__` checks fire under HIP by
// defining the macro during the HIP device pass only. (Host pass under
// hipcc does not set `__HIP_DEVICE_COMPILE__`, so this remains scoped
// to device code.) Value 1 is sufficient — nothing in this codebase
// compares __CUDA_ARCH__ numerically.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
#define __CUDA_ARCH__ 1
#endif

#elif defined(GKYL_HAVE_CUDA)

#include <cuda_runtime.h>

#endif
