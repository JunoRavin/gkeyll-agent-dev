# AMD GPU Port Plan: gkeyll-agent-dev/core/

## Context and Scope

This plan covers porting `gkeyll-agent-dev/core/` to AMD GPUs (HIP/ROCm), targeting AMD MI250X (gfx90a) on the Frontier supercomputer at OLCF. The approach mirrors the cholla porting pattern: lean on hipcc auto-translation, add a single thin runtime shim, and isolate the few places where source-level changes are unavoidable.

**In scope:**
- Build-system changes (configure script, Makefile flags, macros)
- Kernel files (`*_cu.cu`) and the memory/stream wrappers they depend on
- Reduction kernels (CUB → hipCUB)
- Collective communication (NCCL → RCCL)
- Dense BLAS (cuBLAS → rocBLAS) where used outside the deprecated solver path
- Unit tests in `core/unit/` that exercise GPU paths

**Out of scope (this plan):**
- Linear solver infrastructure (`cusolver_ops.cu`, `cudss_ops.cu`) — see §7 for the deprecation rationale
- Simulation apps (`moments/`, `vlasov/`, `gyrokinetic/`, `pkpm/`)
- Performance tuning beyond what's required for correctness

## 0. Guiding Principles

1. **Mirror cholla.** A single toggle in the build system selects HIP vs CUDA. Code stays CUDA-style at the source level; a translation shim header handles the rename to HIP at the include level.
2. **Lean on hipcc.** Most `.cu` files only call CUDA Runtime API and use kernel-launch syntax — these are API-compatible with HIP and need no source change beyond the shim.
3. **Wrap only where APIs genuinely diverge.** Reductions (CUB) and collectives (NCCL) need explicit handling. The CUDA Runtime API and the BLAS-level API translate at the include layer.
4. **Don't rename existing wrapper functions.** `gkyl_cu_malloc`, `gkyl_cu_memcpy`, `gkyl_cu_free`, etc. stay as-is. Internally they dispatch to either CUDA or HIP at compile time.
5. **No premature abstractions.** Every new file or wrapper must justify its existence against the cost of an additional indirection layer.

## 1. Build System

### 1a. New configure script

Add `machines/configure.frontier-gpu.sh`, modeled on `machines/configure.perlmutter.gpu.sh`:

```bash
module load rocm cray-mpich craype-accel-amd-gfx90a cray-hdf5 cray-fftw
export MPICH_GPU_SUPPORT_ENABLED=1
export ROCFFT_RTC_CACHE_PATH=/dev/null

./configure CC=cc GPUCXX=hipcc GPU_ARCH=gfx90a \
    --use-hip=yes --use-rccl=yes --use-mpi=yes \
    --use-cudss=no
```

The split between `CC=cc` (Cray host wrapper) and `GPUCXX=hipcc` for `.cu` files mirrors cholla's `make.host.frontier`. The Cray `cc` wrapper picks up MPI flags automatically when `craype-accel-amd-gfx90a` is loaded.

### 1b. `configure` script extensions

Extend `gkeyll-agent-dev/configure` to recognize:
- `GPU_ARCH` (replaces `CUDA_ARCH` semantically; passed to either nvcc or hipcc)
- `GPUCXX` (separate from `CC` — needed for the dual-compiler model)
- `--use-hip=yes` → writes `USE_HIP=1` to `config.mak`
- `--use-rccl=yes` → writes `USE_RCCL=1` to `config.mak`
- `--rccl-inc`, `--rccl-lib` paths (both default to under `$ROCM_PATH`)
- `ROCM_PATH` propagated to `config.mak`

### 1c. Top-level `Makefile` (canonical home of `USING_NVCC`)

The canonical `USING_NVCC` block lives in the **top-level [`Makefile`](../Makefile)** at lines 75-92, 110, 120-126, 162-175, and 193-199, which exports the relevant variables to all sub-makes. `core/Makefile` is a thin sub-make wrapper and does not own this logic.

Add a sibling `USING_HIPCC` that fires when `$(GPUCXX)` is `hipcc`:

```makefile
ifdef USING_NVCC
  GPU_FLAGS  = -x cu -dc -arch=sm_$(GPU_ARCH) -rdc=true \
               --compiler-options="-fPIC" -Xptxas --disable-optimizer-constants
  GPU_LIBS   = -lcublas -lcusparse
  CFLAGS    += -DGKYL_HAVE_CUDA -DGKYL_HAVE_GPU
  BUILD_DIR  = cuda-build
endif
ifdef USING_HIPCC
  GPU_FLAGS   = --offload-arch=$(GPU_ARCH) -fPIC -fgpu-rdc
  GPU_LDFLAGS = --hip-link -fgpu-rdc
  GPU_LIBS    = -L$(ROCM_PATH)/lib -lamdhip64 -lrocblas
  CFLAGS     += -DGKYL_HAVE_HIP -DGKYL_HAVE_GPU -I$(ROCM_PATH)/include
  BUILD_DIR   = hip-build
endif
```

Two structural notes:
- `-I$(ROCM_PATH)/include` is on global `CFLAGS`, not only `GPU_FLAGS`. Cray `cc` builds .c files that transitively include `gkyl_array.h` → `gkyl_util.h` → `<hip/hip_runtime.h>` (after the gate fix in §3a-bis). Without ROCm headers on host CFLAGS, host TUs fail to compile. Mirrors cholla's `HIPCONFIG = -I$(ROCM_PATH)/include $(shell hipconfig -C)` propagation pattern.
- `-fgpu-rdc` (relocatable device code) is the hipcc analog of `-rdc=true`; it must appear in BOTH the compile and link steps, or symbol resolution fails.

Update the existing NCCL gate at top-level `Makefile:120-126` from `ifdef USING_NVCC` to `ifdef USING_NVCC || ifdef USING_HIPCC` and add a parallel RCCL gate when `USE_RCCL=yes`.

Export the new variables (`USING_HIPCC`, `GPU_FLAGS`, `GPU_LDFLAGS`, `GPU_LIBS`, `BUILD_DIR`, `ROCM_PATH`) at top-level `Makefile:193-199` so sub-makes see them.

### 1d. `core/Makefile-core` (sub-make adjustments)

The `.cu` compilation rule in `core/Makefile-core:68` becomes:
```makefile
%.cu.o: %.cu
	$(GPUCXX) -c $(CFLAGS) $(GPU_FLAGS) $(INCS) $< -o $@
```

The gate that includes `.cu` files in `SRCS` (`core/Makefile-core:12-32`) becomes `USING_NVCC || USING_HIPCC`.

### 1e. `gkeyll/Makefile-gkeyll` and `Makefile_for_ext_C_input`

- [`gkeyll/Makefile-gkeyll:32`](../gkeyll/Makefile-gkeyll#L32) currently links the executable with `${CUDA_LIBS}`. Change to `${GPU_LIBS}` (populated by §1c). Required for the final executable to link on Frontier.
- [`Makefile_for_ext_C_input:40-54`](../Makefile_for_ext_C_input#L40-L54) has its own `USING_NVCC` block for external app builds. Mirror the §1c changes there. Required for downstream Frontier app compilation.

### 1f. `corelinkobjs.mak`

When `USE_RCCL=yes`, no new objects are added — `nccl_comm.cu.o` (or `.c.o`, depending on file extension) handles both backends via the macro switch in §4. When `USE_CUDSS=no` and the AMD path is selected, `cusolver_ops.cu.o` and `cudss_ops.cu.o` are simply not built — see §7.

## 2. Macro Scheme

| Macro | When set | Purpose |
|---|---|---|
| `GKYL_HAVE_CUDA` | `USING_NVCC=yes` | Selects CUDA-specific bodies |
| `GKYL_HAVE_HIP` | `USING_HIPCC=yes` | Selects HIP-specific bodies |
| `GKYL_HAVE_GPU` | either of the above | For code that should run on any GPU (most kernels, most tests) |
| `GKYL_HAVE_NCCL` | `USE_NCCL=yes` (CUDA-only) | Existing — unchanged |
| `GKYL_HAVE_RCCL` | `USE_RCCL=yes` (HIP-only) | New |
| `GKYL_HAVE_CUDSS` | unchanged | NVIDIA-only — stays |

Migration rule for the ~20 existing `#ifdef GKYL_HAVE_CUDA` sites in `core/`:
- Sites calling generic CUDA Runtime API (e.g., `cudaStreamDestroy` in `array.c:59`) → change to `#ifdef GKYL_HAVE_GPU`.
- Sites calling CUDA-specific libraries (cuDSS, cuSolver) → leave as `#ifdef GKYL_HAVE_CUDA`.

## 3. Tier-1 Work — Kernels and Memory

### 3a. GPU runtime translation shim

New header `core/zero/gkyl_gpu_runtime.h`:

```c
#if defined(GKYL_HAVE_HIP)
  #include <hip/hip_runtime.h>
  #define cudaMalloc           hipMalloc
  #define cudaFree             hipFree
  #define cudaMemcpy           hipMemcpy
  #define cudaMemcpyAsync      hipMemcpyAsync
  #define cudaMemset           hipMemset
  #define cudaMallocHost       hipHostMalloc
  #define cudaFreeHost         hipHostFree
  #define cudaStream_t         hipStream_t
  #define cudaStreamCreate     hipStreamCreate
  #define cudaStreamDestroy    hipStreamDestroy
  #define cudaStreamSynchronize hipStreamSynchronize
  #define cudaDeviceSynchronize hipDeviceSynchronize
  #define cudaSuccess          hipSuccess
  #define cudaError_t          hipError_t
  #define cudaGetLastError     hipGetLastError
  #define cudaGetErrorString   hipGetErrorString
  #define cudaMemcpyHostToHost hipMemcpyHostToHost
  #define cudaMemcpyHostToDevice hipMemcpyHostToDevice
  #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
  #define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
  #define cudaDeviceReset      hipDeviceReset
  #define cudaGetDeviceCount   hipGetDeviceCount
  #define cudaSetDevice        hipSetDevice
  #define atomicCAS            atomicCAS
  #define __double_as_longlong __double_as_longlong
  #define __longlong_as_double __longlong_as_double
#elif defined(GKYL_HAVE_CUDA)
  #include <cuda_runtime.h>
#endif
```

(The last three lines are no-ops semantically but document that these intrinsics exist on both platforms under the same names — useful when reviewing the reduction code in §5. The four entries above them — `cudaMemcpyHostToHost`, `cudaDeviceReset`, `cudaGetDeviceCount`, `cudaSetDevice` — were added after the review identified usages outside the deprecated solver path: the enum initializer at [gkyl_util.h:99](../core/zero/gkyl_util.h#L99), `__checkCudaErrors__` at [gkyl_util.h:114](../core/zero/gkyl_util.h#L114), and the device-selection calls at [nccl_comm.c:672-675](../core/zero/nccl_comm.c#L672).)

The shim is included by `gkyl_util.h` (after the gate fix in §3a-bis) and the public headers `gkyl_alloc.h`, `gkyl_array.h`, `gkyl_nccl_comm.h`, `gkyl_nccl_comm_priv.h` so call sites get correct type aliasing automatically — including struct-field declarations that name `cudaStream_t` before any function body executes.

### 3a-bis. Compiler-driver gate fix (silent-miscompile blocker)

The CUDA support block in [`gkyl_util.h:88`](../core/zero/gkyl_util.h#L88) is gated `#ifdef __NVCC__`. This compiler-driver macro is **not** defined under hipcc (which sets `__HIPCC__`). Without intervention, every symbol the block defines — `GKYL_CU_DH`, `GKYL_CU_D`, `GKYL_HAVE_CUDA`, `enum gkyl_cu_memcpy_kind` (172 call sites across `core/`), `GKYL_DEFAULT_NUM_THREADS = 256`, `checkCuda(...)` — silently falls through to the host-only `#else` branch, where `GKYL_DEFAULT_NUM_THREADS = 1` and the enum integer values do not match `hipMemcpy*`. The entire AMD GPU build silently miscompiles.

**Required fix:** Change [`gkyl_util.h:88`](../core/zero/gkyl_util.h#L88) from `#ifdef __NVCC__` to a build-flag gate driven by §2:

```c
#if defined(GKYL_HAVE_GPU)
  // ... was the __NVCC__ block ...
#endif
```

`GKYL_HAVE_GPU` is set by either `USING_NVCC` or `USING_HIPCC` (per §2 and §1c), decoupling the GPU-support gate from the compiler-driver detection.

**Required shim placement:** The translation shim from §3a (`gkyl_gpu_runtime.h`) must be included **at the top of every header that names a GPU type in struct fields or function signatures**:

- [`gkyl_util.h`](../core/zero/gkyl_util.h) — defines `enum gkyl_cu_memcpy_kind` and references `cudaStream_t` in helpers
- [`gkyl_array.h`](../core/zero/gkyl_array.h) — declares `cudaStream_t iostream` in struct `gkyl_array`
- [`gkyl_alloc.h`](../core/zero/gkyl_alloc.h) — propagates types from `gkyl_util.h`
- [`gkyl_nccl_comm.h:16`](../core/zero/gkyl_nccl_comm.h#L16) — declares `cudaStream_t custream` in the public struct
- [`gkyl_nccl_comm_priv.h:49`](../core/zero/gkyl_nccl_comm_priv.h#L49) — same declaration in the private struct

Without the shim included before the struct definition, `cudaStream_t` is undefined under `<rccl/rccl.h>` and the header fails to compile. The §4a edit (which only patches `nccl_comm.c`) is insufficient on its own.

### 3b. Files affected (Tier-1)

- `core/zero/alloc.c` — change `#include <cuda_runtime.h>` to `#include <gkyl_gpu_runtime.h>`. Flip outer `#ifdef GKYL_HAVE_CUDA` to `#ifdef GKYL_HAVE_GPU`. Body works as-is.
- `core/zero/array.c` — same treatment for `cudaStreamDestroy` site at line 59.
- All ~20 `_cu.cu` files in `core/zero/` (array_ops, array_average, dg_bin_ops, dg_basis_ops, dg_array_mask, dg_geom, dg_interpolate, cart_modal_*, nodal_ops, skin_surf_from_ghost, tensor_field_ops, etc.) — change the include. Should be a single find/replace.
- Kernel-launch syntax `kernel<<<grid, block>>>(args)` — works under hipcc unmodified.
- `__global__`, `__device__`, `__host__`, `threadIdx`, `blockIdx`, `blockDim`, `gridDim` — work under hipcc unmodified.
- `atomicAdd` on `double` — supported on gfx90a natively; no change required.

The reduction files (`array_reduce_cu.cu`, `array_dg_reduce_cu.cu`, `array_integrate_cu.cu`) need an additional include change beyond the shim — see §5.

## 4. Collective Communication: RCCL via API-Compatible Swap

RCCL's public API is byte-compatible with NCCL at the symbol level. Verified directly from [ROCm/rccl/src/nccl.h.in](https://github.com/ROCm/rccl/blob/develop/src/nccl.h.in): the header declares `ncclSend`, `ncclRecv`, `ncclAllReduce`, `ncclCommInitRank`, `ncclComm_t`, `ncclResult_t` — same names as NCCL, with one signature difference: stream parameters are typed `hipStream_t` instead of `cudaStream_t`. The shim in §3a aliases that away.

### 4a. Source changes (entire delta)

1. Top of `core/zero/nccl_comm.c`:
   ```c
   #include <gkyl_gpu_runtime.h>   // must come BEFORE rccl.h so cudaStream_t == hipStream_t
   #if defined(GKYL_HAVE_RCCL)
     #include <rccl/rccl.h>
   #elif defined(GKYL_HAVE_NCCL)
     #include <nccl.h>
   #endif
   ```
2. Outer `#ifdef GKYL_HAVE_NCCL` guards in three files become `#if defined(GKYL_HAVE_NCCL) || defined(GKYL_HAVE_RCCL)`:
   - `core/zero/nccl_comm.c:1`
   - `core/zero/multib_comm_conn_nccl.c:4`
   - `core/unit/mctest_nccl_comm.c:3`
3. `core/Makefile` adds `-lrccl` from `$(ROCM_PATH)/lib` when `USE_RCCL=yes`. `USE_NCCL` and `USE_RCCL` are mutually exclusive; `USE_RCCL=yes` requires `USING_HIPCC=yes`.

That's the complete change. The 700+ lines of NCCL collective logic in `nccl_comm.c` (group-call patterns, `g2_nccl_datatype[]` mapping, async-error polling) is reused as-is.

### 4b. Filename and namespace decisions

The file stays `nccl_comm.c`. The function names already use the `nccl_comm` namespace (e.g., `gkyl_nccl_comm_new`); renaming these would force a sweep of every constructor call site for purely cosmetic reasons. Skip.

### 4c. Fallback path (Option A)

If gkeyll ever needs an NCCL feature that RCCL hasn't shipped (newer collective, newer config field), split `nccl_comm.c` into sibling `rccl_comm.c` at that point. Premature splitting now buys nothing and costs ~700 duplicated lines.

## 5. Reductions and Parallel Primitives (Expanded)

This section was substantially expanded from the original draft. The original survey reported "no CUB usage detected" — that was incorrect. CUB is used heavily, and porting reductions is the most architecturally significant part of the AMD work.

### 5a. Current state (NVIDIA path)

Three files use CUB block-level reductions:
- `core/zero/array_reduce_cu.cu` — Min/Max/Sum over array or sub-range
- `core/zero/array_dg_reduce_cu.cu` — DG-component-aware variants
- `core/zero/array_integrate_cu.cu` — Integration via reduction

All three include `<cub/cub.cuh>` and use:
- `cub::BlockReduce<double, BLOCKSIZE>` — block-level reduction primitive
- `cub::Max()`, `cub::Min()`, `cub::Sum()` — classic functor API
- (CUDA 12.9+ only) `::cuda::maximum`, `::cuda::minimum`, `::cuda::std::plus` — newer functor API, gated by `#if CUDART_VERSION > 12090`

Block size: `GKYL_DEFAULT_NUM_THREADS = 256` (defined at `core/zero/gkyl_util.h:105`). 256 is a multiple of both 32 (NVIDIA warp) and 64 (AMD wave) — see §5d.

Hand-rolled `atomicMax_double` / `atomicMin_double` (`array_reduce_cu.cu:23-47`) use `atomicCAS` + `__double_as_longlong` + `__longlong_as_double` to work around the lack of native double atomicMax/Min on both architectures.

For sum: native `atomicAdd(double*, double)` — supported on NVIDIA (CC 6.0+) and AMD gfx90a.

### 5b. AMD analog: hipCUB on rocPRIM

- **rocPRIM**: AMD's native parallel primitives library. Provides `rocprim::block_reduce`, etc. Different namespace, different API. Ships with the `rocm` module.
- **hipCUB**: Header-only wrapper that re-implements CUB's interfaces in the `hipcub::` namespace, on top of rocPRIM. Header: `<hipcub/hipcub.hpp>`. Also ships with the `rocm` module.

`hipcub::BlockReduce`, `hipcub::Max`, `hipcub::Min`, `hipcub::Sum` are the equivalents of their `cub::` counterparts. Internal algorithms differ (rocPRIM has its own block-reduction strategies tuned for wave64), but the API surface and result are identical.

### 5c. Source change strategy

New header `core/zero/gkyl_gpu_reduce.h`:

```cpp
#if defined(GKYL_HAVE_HIP)
  #include <hipcub/hipcub.hpp>
  namespace cub = hipcub;     // namespace alias — clean, scoped, type-safe
#elif defined(GKYL_HAVE_CUDA)
  #include <cub/cub.cuh>
#endif
```

The three reduction files change `#include <cub/cub.cuh>` to `#include <gkyl_gpu_reduce.h>`. Body unchanged. `cub::BlockReduce<double, BLOCKSIZE>` resolves to `hipcub::BlockReduce<double, BLOCKSIZE>` on AMD via the namespace alias.

A namespace alias is preferred over `#define cub hipcub` because:
- It's scoped (only affects code that includes the header)
- It's type-safe (no preprocessor surprises)
- It works correctly with template instantiation and ADL

### 5d. Warp size: 32 (NVIDIA) vs 64 (AMD MI250X)

The CUB / hipCUB BlockReduce abstraction hides warp size from user code. Block-level results are correct regardless. What matters:

**Correctness invariants:**
- `BLOCKSIZE` must be a multiple of warp/wave size. `GKYL_DEFAULT_NUM_THREADS = 256` divides evenly into both 32 and 64. ✓
- User code must NOT assume `threadIdx.x % 32 == 0` is a warp leader. Verified by grep — there are no `__shfl`, `__ballot`, `warpSize`, `WARP_SIZE`, or hardcoded `32` references in any reduction kernel. ✓
- Atomic operations on the final reduction value (post-block-reduce) are warp-size agnostic. ✓

**Performance considerations (NOT correctness):**
- AMD's larger wave (64) means BlockReduce on 256 threads/block uses 4 waves; NVIDIA uses 8 warps. Reduction tree depth differs.
- rocPRIM exposes architecture-specific tunings; hipCUB picks them automatically on gfx90a.
- If Phase 3 reductions show a perf regression vs CUDA on the same problem size, the tuning point is `GKYL_DEFAULT_NUM_THREADS` per architecture. Not a Phase 3 blocker.

### 5e. Atomic operations

All atomic operations used in the reduction code port unchanged:

| Operation | NVIDIA | AMD gfx90a | Status |
|---|---|---|---|
| `atomicAdd(double*, double)` | Native (CC 6.0+) | Native | Works as-is |
| `atomicCAS(unsigned long long int*, ...)` | Supported | Supported | Works as-is |
| `__double_as_longlong` | Intrinsic | Intrinsic | Works as-is |
| `__longlong_as_double` | Intrinsic | Intrinsic | Works as-is |

The hand-rolled `atomicMax_double` / `atomicMin_double` block at `array_reduce_cu.cu:23-47` is unchanged on both platforms.

### 5f. CUDART_VERSION branch

The newer `::cuda::maximum`, `::cuda::minimum`, `::cuda::std::plus` functors are CUDA 12.9+ Standard Library extensions. They don't exist in hipCUB. On AMD, `CUDART_VERSION` is undefined under hipcc, so the `#else` branch (`cub::Max()` / `cub::Min()` / `cub::Sum()`) — which hipCUB does support — fires automatically.

**Verification needed:** Confirm `CUDART_VERSION` is undefined under hipcc on Frontier when no CUDA SDK is on the include path. If it leaks in, extend the guard to `#if defined(CUDART_VERSION) && CUDART_VERSION > 12090 && !defined(GKYL_HAVE_HIP)`.

### 5g. Tests as canary

`ctest_array_reduce.c` and `ctest_array_dg_reduce.c` exercise all three reductions (max, min, sum) in scalar, range-restricted, and DG-component variants. These are the **critical correctness gates** for the warp-size and hipCUB integration. They run immediately after `ctest_array_cu` in the bring-up sequence (§8b) — earlier than in the original plan, because reductions are no longer assumed to be a free auto-translation.

## 6. cuBLAS → rocBLAS

`core/zero/mat.c` directly calls cuBLAS for dense matrix operations: `cublasDgemm`, `cublasDgemm_StridedBatched`, `cublasDgetrfBatched`, `cublasDgetrsBatched`. rocBLAS exposes `rocblas_dgemm`, `rocblas_dgemm_strided_batched`, `rocblas_dgetrf_batched`, `rocblas_dgetrs_batched` with matching signatures.

### 6a. Strategy

Add a thin shim header `core/zero/gkyl_gpu_blas.h` mapping `cublas*` → `rocblas_*` under `GKYL_HAVE_HIP`:

```c
#if defined(GKYL_HAVE_HIP)
  #include <rocblas/rocblas.h>
  #define cublasHandle_t       rocblas_handle
  #define cublasStatus_t       rocblas_status
  #define cublasCreate_v2      rocblas_create_handle
  #define cublasDestroy        rocblas_destroy_handle
  #define cublasDgemm          rocblas_dgemm
  #define cublasDgemm_StridedBatched rocblas_dgemm_strided_batched
  #define cublasDgetrfBatched  rocblas_dgetrf_batched
  #define cublasDgetrsBatched  rocblas_dgetrs_batched
  #define CUBLAS_OP_N          rocblas_operation_none
  #define CUBLAS_OP_T          rocblas_operation_transpose
  #define CUBLAS_STATUS_SUCCESS rocblas_status_success
#elif defined(GKYL_HAVE_CUDA)
  #include <cublas_v2.h>
#endif
```

`mat.c` and `gkyl_mat_priv.h` change `#include <cublas_v2.h>` to `#include <gkyl_gpu_blas.h>`. Body unchanged.

### 6b. Batched LU/solve scope (resolved)

The batched LU functions `cublasDgetrfBatched` and `cublasDgetrsBatched` are wrapped by `cu_nmat_linsolve_lu` ([core/zero/mat.c:600](../core/zero/mat.c#L600)). The public entry point `gkyl_nmat_linsolve_lu_pa` ([core/zero/mat.c:736](../core/zero/mat.c#L736)) is called from core kernels at [dg_bin_ops_cu.cu:355](../core/zero/dg_bin_ops_cu.cu#L355) and [dg_bin_ops_cu.cu:446](../core/zero/dg_bin_ops_cu.cu#L446) (`gkyl_dg_div_op_cu` / `gkyl_dg_div_op_range_cu`). The path is *not* tied to the deprecated solver infrastructure. **rocBLAS port of the batched LU is required**; without it, `ctest_dg_bin_ops` fails to link.

Simulation apps (`vlasov/`, `gyrokinetic/`, `pkpm/`) also call `gkyl_nmat_linsolve_lu_pa` from their `_cu.cu` files. These are out of scope for this porting effort but will benefit automatically once §6a lands.

### 6c. Test gate

`ctest_mat.c` is the verification target.

## 7. Out of Scope: Linear Solver Infrastructure

Linear solver porting is **explicitly out of scope** for this AMD effort.

**Project context:** Gkeyll is planning to deprecate its existing cuSolvers infrastructure (`core/zero/cusolver_ops.cu`) because NVIDIA cut support for the cuSolverRf-based factorization pattern Gkeyll was using. The cuDSS path (`core/zero/cudss_ops.cu`) is also deprecated for AMD use because cuDSS has no AMD equivalent (rocSPARSE provides primitives but no end-to-end sparse direct solver matching cuDSS's interface).

**Disposition for this port:**
- Keep `cusolver_ops.cu`, `cudss_ops.cu`, `gkyl_cusolver_ops.h`, `gkyl_cudss_ops.h`, `gkyl_culinsolver_ops.h` gated behind `#ifdef GKYL_HAVE_CUDA`. Do not introduce a HIP path.
- Frontier builds run with `--use-cudss=no` (default) and the cuSolver translation unit simply isn't compiled.
- Tests `ctest_cusolver.cu`, `ctest_cudss.cu`, and the CUDA-gated branches of `ctest_linsolvers.c` are CUDA-only and skipped on Frontier.
- A future replacement (e.g., SuperLU_DIST integration via `install-deps/build-superlu_dist.sh`) is tracked separately and is not part of this porting effort.

This decision dramatically reduces scope and risk. The original Phase 6 in the plan is removed.

## 8. Unit Tests

### 8a. Existing tests, classified

| Class | Tests | Action |
|---|---|---|
| Paired `_cu.cu` | ctest_alloc_cu, ctest_array_cu, ctest_basis_cu, ctest_range_cu, ctest_rect_grid_cu, ctest_struct_of_arrays_cu | Compile with hipcc via shim; gate host-side blocks with `#ifdef GKYL_HAVE_GPU`. No new files. |
| GPU through C API | ctest_array_ops, ctest_array_reduce, ctest_array_dg_reduce, ctest_array_integrate, ctest_dg_bin_ops, ctest_array_average, ctest_dg_array_mask, ctest_struct_of_arrays, ctest_tensor_field_ops | No source changes — verify they pass on AMD. ctest_array_reduce + ctest_array_dg_reduce are the **reduction correctness gate**. |
| Collectives | mctest_nccl_comm | Gate change to `#if defined(GKYL_HAVE_NCCL) \|\| defined(GKYL_HAVE_RCCL)`. Same binary on both platforms. |
| Linear solvers (out of scope) | ctest_cusolver, ctest_cudss, ctest_linsolvers | Stay NVIDIA-only via `#ifdef GKYL_HAVE_CUDA`. Skipped on AMD. |
| BLAS | ctest_mat | Should pass on AMD via the rocBLAS shim. |

### 8b. AMD bring-up sequence (ordered)

Run on Frontier in this order. A failure halts the sequence and indicates which subsystem has the bug:

1. `ctest_alloc_cu` — verifies `hipMalloc` / `hipFree` and shim correctness
2. `ctest_array_cu` — verifies kernel launch and `__global__` semantics
3. `ctest_array_ops` — verifies Tier-1 arithmetic kernels (add, multiply, scale, accumulate)
4. **`ctest_array_reduce`** — verifies hipCUB BlockReduce and warp-size correctness (canary for §5)
5. **`ctest_array_dg_reduce`** — verifies DG-component reductions
6. `ctest_array_integrate` — verifies integration (third hipCUB user)
7. `ctest_dg_bin_ops` — verifies the heavier DG kernels
8. `ctest_mat` — verifies rocBLAS shim
9. `mctest_nccl_comm` (single node, 2 ranks, linked against RCCL) — verifies collective communication

A pass on all nine = green light to start porting the simulation apps in a later phase.

### 8c. Coverage gap closure: new minimal smoke tests

Five core kernel files have no `core/unit/` GPU exercise — a regression on AMD would not surface until simulation-app shakedown. We close the gap directly rather than rely on the downstream phase:

| Kernel file | New test | Purpose |
|---|---|---|
| [`cart_modal_hybrid_cu.cu`](../core/zero/cart_modal_hybrid_cu.cu) | extend [`ctest_basis_cu.cu`](../core/unit/ctest_basis_cu.cu) | Construct hybrid basis via `_cu_dev_new`; assert non-null and one device-side `eval` round-trip matches host |
| [`cart_modal_gkhybrid_cu.cu`](../core/zero/cart_modal_gkhybrid_cu.cu) | extend [`ctest_basis_cu.cu`](../core/unit/ctest_basis_cu.cu) | Same pattern for gkhybrid basis |
| [`dg_geom_cu.cu`](../core/zero/dg_geom_cu.cu) | new `ctest_dg_geom_cu.cu` | `gkyl_dg_geom_new_from_host(..., true)` then assert struct fields populated and one geom value matches host |
| [`dg_interpolate_cu.cu`](../core/zero/dg_interpolate_cu.cu) | new `ctest_dg_interpolate_cu.cu` | Construct interpolator with `use_gpu=true`, advance one input array, compare against host result on a small grid |
| [`nodal_ops_cu.cu`](../core/zero/nodal_ops_cu.cu) | new `ctest_nodal_ops_cu.cu` | Round-trip nodal→modal→nodal on GPU, assert max-norm diff < tolerance |

Each new test follows the same minimal pattern: constructor invocation, one kernel call, one sanity assertion. Cost is small (≈30-50 lines per test) and the bring-up sequence in §8b extends to nine + five = fourteen tests. New tests for `dg_geom`, `dg_interpolate`, `nodal_ops` slot in after `ctest_dg_bin_ops` (position 7); the basis constructor sanity checks run alongside the existing `ctest_basis_cu` step (which moves earlier in the sequence as a Phase 2 verification).

> **Deferred to a later phase.** The `_cu.cu` files for these five kernels all compile cleanly under hipcc as part of Phase 4 (verified in `hip-build/core/zero/`), but authoring the five new smoke tests has been pushed out — likely paired with the simulation-app shakedown phase, where these kernels first see end-to-end exercise. The compile gate is sufficient to keep Phase 4 unblocked; the runtime gap remains until the smoke tests land.

## 9. Frontier Build & Test Recipe

```bash
git clone <gkeyll-agent-dev repo>
cd gkeyll-agent-dev
source machines/configure.frontier-gpu.sh    # loads modules + invokes ./configure
make core -j 16
make core-unit -j 16

# Inside an allocation (substitute your project ID):
salloc -A <PROJECT_ID> -t 00:30:00 -N 1 --gpus 1
srun -n 1 -c 8 --gpus-per-task=1 hip-build/core/unit/ctest_alloc_cu
srun -n 1 -c 8 --gpus-per-task=1 hip-build/core/unit/ctest_array_cu
srun -n 1 -c 8 --gpus-per-task=1 hip-build/core/unit/ctest_array_ops
srun -n 1 -c 8 --gpus-per-task=1 hip-build/core/unit/ctest_array_reduce       # canary
srun -n 1 -c 8 --gpus-per-task=1 hip-build/core/unit/ctest_array_dg_reduce
srun -n 1 -c 8 --gpus-per-task=1 hip-build/core/unit/ctest_array_integrate
srun -n 1 -c 8 --gpus-per-task=1 hip-build/core/unit/ctest_dg_bin_ops
srun -n 1 -c 8 --gpus-per-task=1 hip-build/core/unit/ctest_mat
srun -n 2 -c 8 --gpus-per-task=1 hip-build/core/unit/mctest_nccl_comm
```

## 10. Phasing

| Phase | Deliverable | Gate to next phase |
|---|---|---|
| 0 | cholla baseline build + smoke test on Frontier | (separately tracked) |
| 1 | `configure.frontier-gpu.sh`; `USING_HIPCC` in top-level `Makefile` + `Makefile-gkeyll` + `Makefile_for_ext_C_input`; `BUILD_DIR=hip-build`; `GKYL_HAVE_HIP` / `GKYL_HAVE_GPU` macros; `gkyl_gpu_runtime.h` shim; `gkyl_util.h:88` gate fix per §3a-bis; shim included in `gkyl_util.h`, `gkyl_array.h`, `gkyl_alloc.h`, `gkyl_nccl_comm.h`, `gkyl_nccl_comm_priv.h` | Empty build links cleanly |
| 2 | `alloc.c` + `array.c` + `_cu.cu` shim conversions | `ctest_alloc_cu`, `ctest_array_cu` pass |
| 3 | `gkyl_gpu_reduce.h` + reduction file conversions | `ctest_array_reduce`, `ctest_array_dg_reduce`, `ctest_array_integrate` pass |
| 4 | All other `_cu.cu` files compile with hipcc. (§8c smoke tests deferred — see note in §8c.) | `ctest_array_ops` passes; the §6a-tracked LU-dependent paths in `ctest_dg_bin_ops` and `ctest_array_average` remain failing on AMD until Phase 5 ports `gkyl_nmat_cu_dev_new`/`gkyl_nmat_linsolve_lu_pa` to rocBLAS — both call sites assert-stub through `mat.c` under HIP today. |
| 5 | `gkyl_gpu_blas.h` + `mat.c` conversion | `ctest_mat` passes |
| 6 | RCCL via macro switch | `mctest_nccl_comm` passes single-node |

(No Phase 7 — linear solver porting is out of scope per §7.)

## 11. Risks and Open Questions

1. **`-fgpu-rdc` correctness on Frontier.** hipcc's relocatable-device-code linking is fragile if any TU is built without it. All `.cu` compilations and the final link must agree. Will surface as undefined-symbol errors at link time, not silent miscompiles.
2. **Cray `cc` wrapper interactions with hipcc.** OLCF recommends compiling host code with `cc` and device code with `hipcc`. Need to verify that `cc` correctly picks up MPI flags when `craype-accel-amd-gfx90a` is loaded, AND that hipcc-compiled `.o` files link cleanly with `cc`-compiled `.o` files. cholla uses this pattern successfully ([cholla/builds/make.host.frontier:3-5](../../cholla/builds/make.host.frontier#L3-L5)) so it's known to work.
3. **`CUDART_VERSION` leakage under hipcc.** If somehow defined, the CUDA-12.9+ functors are picked and hipCUB doesn't have them. See §5f for the defensive guard.
4. **`hipHostMalloc` flags differ from `cudaMallocHost`.** Default flags differ subtly; in practice `hipHostMalloc(ptr, size)` matches `cudaMallocHost(ptr, size)` semantics. Verify in Phase 2 by checking that pinned-host I/O streams behave correctly.
5. **GPU-aware MPI on Frontier.** Cray MPICH with `MPICH_GPU_SUPPORT_ENABLED=1` is the supported path. The existing `gkyl_mpi_comm.c` uses `MPI_Send`/`MPI_Recv` on raw pointers — should "just work" if the pointer is a device pointer and the env var is set. Verify in Phase 6.
6. **`hipCUB` header location.** On Frontier, the path may be `<hipcub/hipcub.hpp>` or `<hipcub.hpp>` depending on the ROCm version. Verify on the actual build node before Phase 3.
(Risks #6 "batched LU scope" and #7 "rocBLAS type names" from the pre-review draft were resolved during the audit and are folded into §6a / §6b. The original "coverage gap" risk was promoted to a deliverable in §8c.)
