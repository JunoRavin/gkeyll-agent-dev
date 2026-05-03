# AMD Port — Phase 2 Handoff

This file captures Phase 2 completion state and the Phase 3 plan, so a fresh
Claude session can pick up cleanly.

Source plan: [`amd_port_plan.md`](amd_port_plan.md). Phase 1 state:
[`phase1_handoff.md`](phase1_handoff.md). Read plan §3a, §3b, §5, §8 before
starting Phase 3.

---

## Environment

- Host: `login1` of the Frontier dev sub-system (30 nodes `odo[01-30]`,
  partition `batch`, qos `normal`, 12 h max walltime).
- Account: `fus183`. Storage: `$HOME = /ccsopen/home/junoravin` (work happens
  here, per user direction — not on `/gpfs/wolf2/...`).
- Working dir: `/autofs/nccsopen-svm1_home/junoravin/working_dev/`. Source tree
  under `gkeyll-agent-dev/`. Reference cholla port under `cholla/`.
- Toolchain: `PrgEnv-amd` + `rocm/6.2.4` + `craype-accel-amd-gfx90a` +
  `cray-mpich` + `cray-libsci` (libsci_amd). hipcc, hipCUB, RCCL, rocBLAS all
  present at `/opt/rocm-6.2.4`.
- Permission allowlist already configured at `.claude/settings.json`.
- Bash tool calls do **not** persist module loads. Every build invocation
  must prefix `module load PrgEnv-amd rocm/6.2.4 craype-accel-amd-gfx90a
  cray-mpich cray-libsci` (and export `MPICH_GPU_SUPPORT_ENABLED=1` plus
  `LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH`) inline. Sourcing
  `machines/configure.frontier-gpu.sh` once produces a stable `config.mak`
  but does not survive Bash sessions.

---

## Phase 2 — Complete

**Gate met:** `ctest_alloc` (6/6) and `ctest_array` (11/11) pass on a Frontier
compute node, including device-side tests `cu_malloc`, `cu_malloc_array`,
`mem_buff_dev`, `cu_array_base`, and `cu_array_dev_kernel`. The last of those
actually launches a HIP kernel on gfx90a — proves the shim, the kernel-launch
syntax, the `-fgpu-rdc` link, and gpu-aware MPI plumbing all work end-to-end.

```text
hip-build/core/libg0core.so  25 MB
  291 *.c.o objects      (host code via cc; ker/ overrides via hipcc -x hip)
   14 *.cu.o objects     (zero/_cu.cu files compiled via hipcc + -fgpu-rdc)
    6 *.cu.o objects     (unit/ctest_*_cu.cu device companions)
  device-link via hipcc --hip-link -fgpu-rdc, host-link clean
```

The handoff plan was wrong about test executable names: the gates run as
`hip-build/core/unit/ctest_alloc` and `ctest_array` (NOT `_cu`-suffixed). The
`_cu.cu` files are device-side companions linked into `libg0core.so` and
exercised through the host-side `.c` test driver.

### Files created in Phase 2

None. Phase 2 was a pure conversion phase: existing files got new gate
macros, the shim got tightened, and the build system grew a HIPCC link
branch.

### Files modified

- `core/zero/alloc.c` — outer `#ifdef GKYL_HAVE_CUDA` flipped to
  `#ifdef GKYL_HAVE_GPU`; `#include <cuda_runtime.h>` replaced with
  `#include <gkyl_gpu_runtime.h>`. Comments updated.
- `core/zero/array.c` — three `GKYL_HAVE_CUDA` sites flipped to
  `GKYL_HAVE_GPU` (line 58 destructor, line 225 clone, line 261
  outer-block).
- `core/zero/gkyl_array.h:30` — `iostream` field gate flipped so the HIP
  path gets `cudaStream_t` (= `hipStream_t` via the shim) instead of
  falling through to the `int iostream` non-GPU stub.
- `core/zero/gkyl_alloc.h:147` — `gkyl_cu_memcpy_async` declaration gate
  flipped so its `cudaStream_t stream` parameter matches the
  HIP-build call sites in `array.c`.
- `core/zero/gkyl_gpu_runtime.h` — restructured per "header strategy"
  below. Two header-selection arms based on `__cplusplus`, `extern "C++"`
  wrap on the C++ side, `cudaMallocHost(p, s)` widened to function-style
  macro that fills in `hipHostMalloc`'s `flags` arg with `0`.
- 17 paired host `.c` files in `core/zero/` got `GKYL_HAVE_CUDA` → 
  `GKYL_HAVE_GPU` swept across them with sed:
  `array_average.c`, `array_ops.c`, `cart_modal_gkhybrid.c`,
  `cart_modal_hybrid.c`, `cart_modal_serendip.c`, `cart_modal_tensor.c`,
  `dg_array_mask.c`, `dg_basis_ops.c`, `dg_bin_ops.c`, `dg_geom.c`,
  `dg_interpolate.c`, `eval_on_nodes.c`, `nodal_ops.c`,
  `skin_surf_from_ghost.c`, `tensor_field.c`, `tensor_field_ops.c`,
  `util.c`. Deferred files (`array_reduce.c`, `array_dg_reduce.c`,
  `array_integrate.c`, `mat.c`) intentionally kept on `GKYL_HAVE_CUDA`
  so their assert-stubs satisfy any link references until Phases 3/5.
- 5 priv headers swept the same way:
  `gkyl_array_average_priv.h`, `gkyl_dg_array_mask_priv.h`,
  `gkyl_dg_basis_ops_priv.h`, `gkyl_dg_interpolate_priv.h`,
  `gkyl_skin_surf_from_ghost_priv.h`. Deferred priv headers
  (`gkyl_array_reduce_priv.h`, `gkyl_array_dg_reduce_priv.h`,
  `gkyl_mat_priv.h`) keep `GKYL_HAVE_CUDA`.
- `core/zero/gkyl_skin_surf_from_ghost_priv.h` — dropped `GKYL_CU_D`
  annotation from `skin_surf_from_ghost_choose_kernel`. See "API design
  finding" below.
- `core/unit/ctest_alloc.c`, `core/unit/ctest_array.c` —
  `GKYL_HAVE_CUDA` → `GKYL_HAVE_GPU` so the GPU-side tests are exercised
  under HIP. Without this they silently no-op (the binary builds and
  passes 3 host tests instead of 6/11).
- `Makefile` (top level) — added `-x hip` to `GPU_FLAGS` on the HIP path.
  See "kernel-override `.c` files need `-x hip`" below.
- `core/Makefile-core` — five edits:
  1. Widened the `.cu` SRCS gate to include a `USING_HIPCC` arm that
     `find`s `*.cu` under `$(SRC_DIRS)` and `$(UNIT_DIRS)`, with
     `filter-out` exclusions for the deferred files.
  2. Added a `USING_HIPCC` arm to `UNIT_CU_SRCS` mirroring the NVCC arm
     minus `ctest_cusolver.cu`/`ctest_cudss.cu`.
  3. Widened the kernel-override `.c.o` block (lines 92-116) to fire
     under either `USING_NVCC` or `USING_HIPCC`, and rewrote the recipes
     to use unified `$(GPUCXX) $(CFLAGS) $(GPU_FLAGS) $(INCS) -c $< -o $@`.
     `GPUCXX=$(CC)=nvcc` and `GPU_FLAGS=NVCC_FLAGS` on the NVCC path so
     the existing CUDA build is unchanged.
  4. Added `else ifdef USING_HIPCC` branch to the libg0core.so link rule
     that drives the link through `$(GPUCXX) -shared $(GPU_LDFLAGS)
     -fPIC` (i.e. hipcc with `--hip-link -fgpu-rdc`) and pulls in BOTH
     `CORE_LINK_OBJS` and `CORE_LINK_CU_OBJS`.

### Plan deviations worth folding into `amd_port_plan.md`

Five things surfaced during Phase 2 that the plan-as-written didn't capture
exactly. All are folded in below; mirror these into the plan document:

1. **Header-strategy refinement** (plan §3a). The original shim
   unconditionally `#include <hip/hip_runtime.h>`. That works for `.c`
   under Cray cc but breaks `_cu.cu` compilation: `hip_runtime.h`
   transitively pulls in libstdc++ (`<memory>` → `<bits/unique_ptr.h>`),
   and every `_cu.cu` file wraps Gkyl C-header includes in
   `extern "C" {}`, which poisons the surrounding C linkage when the
   shim is reached transitively. Two complementary fixes:
   - In C TUs, include only `<hip/hip_runtime_api.h>` — the C-clean host
     runtime API. The C++ template helpers in
     `amd_detail/host_defines.h` are gated on `__HIP__` (set only by
     hipcc), so a pure-C TU compiled by Cray cc never sees them. This
     keeps Gkeyll C code free of any libstdc++ leakage.
   - In C++ TUs, include the full `<hip/hip_runtime.h>` (needed for
     device intrinsics: `threadIdx`, `blockIdx`, `blockDim`, `gridDim`,
     atomic functions, math helpers — none of which are in the api
     header) but wrap it in `extern "C++"` so the C++ template content
     parses correctly even when the shim is reached from inside an
     `extern "C" {}` block.

2. **`cudaMallocHost` arity mismatch** (plan §11 risk #4 — confirmed real).
   `hipHostMalloc(void** ptr, size_t size, unsigned int flags)` requires
   3 args; CUDA's `cudaMallocHost(void** ptr, size_t size)` takes 2.
   The shim's symbol-rename `#define cudaMallocHost hipHostMalloc` thus
   produces a "too few arguments" error at every Gkeyll call site. Fix:
   widen the macro to `#define cudaMallocHost(ptr, size)
   hipHostMalloc((ptr), (size), 0)`. `hipHostMallocDefault == 0` matches
   `cudaMallocHost`'s default. This works because Gkeyll only ever
   calls `cudaMallocHost` as a function (never takes its address); the
   function-style macro is fine here.

3. **Kernel-override `.c` files need `-x hip`** (new — fold into plan §1c).
   hipcc auto-enables HIP mode (defining `__HIP__`, taking
   `__device__`/`__global__` annotations seriously) for `.cu`, `.cpp`,
   `.cxx` extensions but NOT for `.c`. The kernel-override block in
   `core/Makefile-core` compiles `ker/array_average/*.c`,
   `ker/bin_op/*.c`, etc. with hipcc — for the device-side
   `GKYL_CU_DH`-annotated kernel functions to land in device bitcode,
   `__HIP__` must be set, which means hipcc must be in HIP mode.
   Adding `-x hip` to `GPU_FLAGS` on the HIP path forces it. Without
   this, the device link fails with a wall of "undefined hidden symbol"
   errors for every `gkyl_array_average_*`, `binop_div_copy_sol`, etc.
   The `-x hip` flag is sticky — it forces HIP mode for the next input
   file in the invocation. Since each rule processes one file, this is
   safe.

4. **HIPCC link branch must drive the lib link, not Cray cc** (new — fold
   into plan §1d). With `-fgpu-rdc` (relocatable device code), each
   `.cu.o` and each `-x hip`-compiled `.c.o` carries unresolved
   device-side symbols. Resolving them requires a device-link step
   driven by `hipcc --hip-link -fgpu-rdc`. Cray cc cannot do this. The
   fix is a third arm in the `$(LLIB)` recipe: `else ifdef USING_HIPCC`
   that calls `$(GPUCXX) -shared $(GPU_LDFLAGS) -fPIC -o $@
   ${CORE_LINK_OBJS} ${CORE_LINK_CU_OBJS} -Wl,-soname,libg0core.so ...`
   (with the rest of the host-side libs unchanged). This mirrors
   cholla's `make.host.frontier` recipe.

5. **Phase 1's gate-flip pattern needs to extend to ALL paired host
   `.c` files and Phase 2 priv headers** (fold into plan §3b). The
   plan listed `alloc.c` and `array.c` explicitly; in practice any
   host `.c` whose `_cu.cu` companion is now compiling under HIP must
   have its `#ifdef GKYL_HAVE_CUDA` and `#ifndef GKYL_HAVE_CUDA`
   gates flipped to `GKYL_HAVE_GPU`. Otherwise:
   - `#ifndef GKYL_HAVE_CUDA` stubs (the assert-false fallbacks) get
     compiled under HIP alongside the real `_cu.cu` impls → duplicate
     symbols at link.
   - `#ifdef GKYL_HAVE_CUDA` gated host code that calls into the `_cu`
     functions → calls dropped entirely under HIP, breaking GPU
     functionality even though the device code is present.
   Same applies to priv headers that declare the `_cu` functions —
   their gates must also flip, or the host `.c` callers see implicit
   declarations and fail to type-check.

### How to rebuild from scratch

```bash
cd /autofs/nccsopen-svm1_home/junoravin/working_dev/gkeyll-agent-dev

# One-time: write config.mak. (Sub-shell — module loads do not persist.)
source machines/configure.frontier-gpu.sh

# Build (modules MUST be loaded in the same Bash invocation):
module load PrgEnv-amd rocm/6.2.4 craype-accel-amd-gfx90a cray-mpich cray-libsci
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

rm -rf hip-build
make core -j 8        # -> hip-build/core/libg0core.so
make core-unit -j 8   # -> hip-build/core/unit/ctest_*

# Run gate tests (compute node, 5-min reservation is plenty):
srun -A fus183 -p batch -N 1 -n 1 -c 8 --gpus-per-task=1 -t 5:00 \
    hip-build/core/unit/ctest_alloc
srun -A fus183 -p batch -N 1 -n 1 -c 8 --gpus-per-task=1 -t 5:00 \
    hip-build/core/unit/ctest_array
```

---

## API design finding (read before Phase 3)

A class of issues that nvcc tolerates and hipcc does not, surfaced first in
[`gkyl_skin_surf_from_ghost_priv.h`](gkeyll-agent-dev/core/zero/gkyl_skin_surf_from_ghost_priv.h):

- The header declares a host-callable wrapper
  `skin_surf_from_ghost_choose_kernel_cu(...)` (impl in `_cu.cu`) and a
  `static GKYL_CU_D` (= `__device__`) helper
  `skin_surf_from_ghost_choose_kernel(...)` whose body calls the
  host-wrapper inside `if (use_gpu) {...}`.
- The static helper is **only ever called from
  `skin_surf_from_ghost.c:28`** (host code). Never from device code.
- nvcc accepted this because `static __device__` functions are
  type-checked lazily — an uncalled-from-device static function is
  emitted nowhere and its dead `__device__` → `__host__` calls are
  ignored.
- hipcc/clang does host/device-aware overload resolution **eagerly**.
  From a `__device__` function context, an implicitly-`__host__` callee
  is "no viable overload", and the diagnostic is the misleading
  "no matching function for call to ...".

**Fix applied:** dropped the `GKYL_CU_D` annotation. The function is
host-only by intent; the annotation was leftover.

**Likely scope beyond Phase 2:** there may be other priv-header static
dispatchers of this shape — "static helper that picks CPU vs GPU kernel
table based on a `use_gpu` flag" — over-decorated with `GKYL_CU_D`.
A `grep -B1 'static.*choose_kernel\|static.*pick_kernel' core/zero/*priv*.h`
sweep against the call-site list (which is always in the corresponding
`.c` file, not `.cu`) is worth doing once before each subsequent phase.
The lookup tables in `gkyl_cart_modal_serendip_priv.h` and similar
(static `__device__` arrays of function pointers indexed by basis dim
and poly order) are the *correct* use of `GKYL_CU_D` — they ARE
consumed by device code.

---

## Phase 3 — Next

**Goal:** the three reduction `_cu.cu` files compile under hipcc on
hipCUB, and the resulting `libg0core.so` passes the reduction unit tests.

**Gate:** `ctest_array_reduce`, `ctest_array_dg_reduce`,
`ctest_array_integrate` pass on a compute node.

### Work items (per plan §5)

1. **Add the hipCUB shim header**
   `core/zero/gkyl_gpu_reduce.h` — model after [`gkyl_gpu_runtime.h`](gkeyll-agent-dev/core/zero/gkyl_gpu_runtime.h):

   ```c
   #pragma once
   #if defined(GKYL_HAVE_HIP)
     // namespace alias is preferred over #define cub hipcub:
     // scoped, type-safe, plays correctly with template instantiation.
     #include <hipcub/hipcub.hpp>
     namespace cub = hipcub;
   #elif defined(GKYL_HAVE_CUDA)
     #include <cub/cub.cuh>
   #endif
   ```

   Verify on Frontier: the include path is `<hipcub/hipcub.hpp>`
   (plan §11 risk #6 said this might be `<hipcub.hpp>` on some ROCm
   builds; confirm with `ls /opt/rocm-6.2.4/include/hipcub/`).

2. **Convert the three reduction files**:
   - `core/zero/array_reduce_cu.cu`: change `#include <cub/cub.cuh>` →
     `#include <gkyl_gpu_reduce.h>`. Body unchanged.
   - `core/zero/array_dg_reduce_cu.cu`: same.
   - `core/zero/array_integrate_cu.cu`: same.

3. **Flip the host-side `.c` and priv-header gates** for these three.
   They were intentionally left on `GKYL_HAVE_CUDA` in Phase 2 to keep
   the assert-stubs in scope. Now the `_cu.cu` impls are compiling, so:
   - `core/zero/array_reduce.c`: `GKYL_HAVE_CUDA` → `GKYL_HAVE_GPU`.
   - `core/zero/array_dg_reduce.c`: same.
   - `core/zero/array_integrate.c`: same.
   - `core/zero/gkyl_array_reduce_priv.h`: same.
   - `core/zero/gkyl_array_dg_reduce_priv.h`: same.
   - The unit tests `core/unit/ctest_array_reduce.c`,
     `ctest_array_dg_reduce.c`, `ctest_array_integrate.c` (if any of
     them have a GKYL_HAVE_CUDA gate — check before flipping).

4. **Remove the deferred files from `HIP_DEFER_REDUCE`** in
   `core/Makefile-core`. After this list is empty for reductions; you
   can drop the variable if it has no remaining entries.

5. **Sanity-check `CUDART_VERSION` is undefined under hipcc on Frontier**
   (plan §5f). The reduction code has a `#if CUDART_VERSION > 12090`
   block that selects newer CUB functors (`::cuda::maximum`, etc.) which
   hipCUB does not provide. If `CUDART_VERSION` leaks in under hipcc,
   the wrong branch fires. Quick check:
   `hipcc -E -dM -x hip - </dev/null | grep -i CUDART`. Should produce
   nothing. If something appears, harden the guard to
   `#if defined(CUDART_VERSION) && CUDART_VERSION > 12090 &&
   !defined(GKYL_HAVE_HIP)`.

6. **Build the unit tests**: `make core-unit -j 8`.

7. **Run the gate tests** on a compute node:
   ```bash
   srun -A fus183 -p batch -N 1 -n 1 -c 8 --gpus-per-task=1 -t 5:00 \
       hip-build/core/unit/ctest_array_reduce
   srun -A fus183 -p batch -N 1 -n 1 -c 8 --gpus-per-task=1 -t 5:00 \
       hip-build/core/unit/ctest_array_dg_reduce
   srun -A fus183 -p batch -N 1 -n 1 -c 8 --gpus-per-task=1 -t 5:00 \
       hip-build/core/unit/ctest_array_integrate
   ```

### Likely fall-out to handle in Phase 3

- **Warp size 32 (NVIDIA) vs wave 64 (AMD MI250X)**: `BLOCKSIZE` in the
  reduction kernels is `GKYL_DEFAULT_NUM_THREADS = 256`, which divides
  cleanly into both. Plan §5d argues this is correctness-safe; verify by
  passing `ctest_array_reduce`. If a reduction returns wrong values,
  start by checking for any hardcoded `32` / `warpSize` / `__shfl` calls
  in the kernels. (Plan grep claimed none exist; trust but verify.)

- **`hipCUB` template instantiation under `extern "C++"`**: the shim's
  C++ side already wraps in `extern "C++"`, so `hipcub::BlockReduce`
  templates should parse cleanly. If you hit linkage errors that blame
  template name mangling, the suspect is double-wrapping `extern "C++"`
  inside an existing `extern "C"`; check the `_cu.cu` file include order.

- **Device-link size**: adding three more `.cu.o` objects with
  hipCUB-instantiated templates may push the libg0core.so device-link
  step past some hipcc/lld memory limit. cholla hits this around
  large kernel counts; if it surfaces, the workaround is `-fgpu-rdc`
  paired with `-fno-gpu-rdc` for specific TUs that don't need
  cross-TU device symbols — but that's a Phase 3-blocker fix, not
  a routine knob.

- **API design audit**: before flipping the priv headers, grep them for
  the `static GKYL_CU_D ... choose_kernel ... if (use_gpu) ... cu(...)`
  pattern. Any matches need the same `GKYL_CU_D` removal that
  `skin_surf_from_ghost_priv.h` got. Reductions don't use the
  dispatcher pattern (they call directly into a single kernel), so
  this is unlikely to surface here, but worth one minute of grep.

### Files NOT to touch in Phase 3

- `core/zero/mat.c` (cuBLAS) — Phase 5.
- `core/zero/nccl_comm.c` and `multib_comm_conn_nccl.c` — Phase 6.
- `core/zero/cusolver_ops.cu`, `core/zero/cudss_ops.cu` — out of scope
  (plan §7).
- `core/unit/ctest_basis_cu.cu` — has a Phase 4 stack-frame issue
  (`ker_dev_cu_ser_2d` exceeds AMDGPU's 131 056-byte per-function
  limit). Already deferred via `HIP_DEFER_UNIT` in
  `core/Makefile-core`. Phase 4 work; not Phase 3.

### Where to find things

- Phase 2 shim (host-only mode): [`gkeyll-agent-dev/core/zero/gkyl_gpu_runtime.h`](gkeyll-agent-dev/core/zero/gkyl_gpu_runtime.h)
- Top-level Makefile USING_HIPCC block: [`gkeyll-agent-dev/Makefile`](gkeyll-agent-dev/Makefile) (lines ~74-131)
- Sub-make: [`gkeyll-agent-dev/core/Makefile-core`](gkeyll-agent-dev/core/Makefile-core)
- Frontier configure: [`gkeyll-agent-dev/machines/configure.frontier-gpu.sh`](gkeyll-agent-dev/machines/configure.frontier-gpu.sh)
- cholla reference (Frontier HIP recipe): [`cholla/builds/make.host.frontier`](cholla/builds/make.host.frontier)
- Phase 1 handoff (for context): [`phase1_handoff.md`](phase1_handoff.md)

### Deferred from Phase 2 (track separately)

- `unit/ctest_basis_cu.cu` — `ker_dev_cu_ser_2d` exceeds AMDGPU's
  per-function 131 056-byte stack limit (135 184 reported). Needs
  spilling to LDS, hoisting to constant memory, or refactoring the
  kernel to avoid large local arrays. Phase 4 work per plan §10
  (basis tests are Phase 4 deliverables).
- Audit of remaining `GKYL_CU_D static ... choose_kernel` dispatchers in
  priv headers — see "API design finding" above. Best done before each
  subsequent phase as a 60-second grep, not as a single bulk sweep.
