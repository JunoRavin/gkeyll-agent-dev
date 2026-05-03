# AMD Port — Phase 1 Handoff

This file captures Phase 1 completion state and the Phase 2 plan, so a fresh
Claude session can pick up cleanly.

Source plan: [`amd_port_plan.md`](amd_port_plan.md). Read §3a, §3a-bis, §3b
before starting Phase 2.

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
- Permission allowlist already configured at `.claude/settings.json` —
  `make`, `module`, `salloc -A fus183 *`, `srun -A fus183 *`, `sbatch -A fus183 *`,
  `squeue/sinfo/scontrol/sacct/scancel/sacctmgr show`, `rm -rf hip-build`,
  `rm -rf cuda-build`, `source machines/configure.frontier-gpu.sh*`,
  `source machines/mkdeps.frontier.sh*`, `hipcc --version`, `hipconfig:*`,
  `ldd:*`. Long-running commands won't prompt.

---

## Phase 1 — Complete

**Gate met:** Empty HIP build links cleanly.

```text
hip-build/core/libg0core.so  9.8 MB
  291 *.c.o objects
    0 *.cu.o objects (Phase 2 enables these)
  ldd: links libamdhip64.so.6, libsci_amd.so.6, libmpi_amd.so.12 — clean
```

### Files created

- `gkeyll-agent-dev/core/zero/gkyl_gpu_runtime.h` — runtime translation shim
  per plan §3a. Aliases `cuda*` → `hip*` when `GKYL_HAVE_HIP` is defined.
- `gkeyll-agent-dev/machines/configure.frontier-gpu.sh` — module loads + the
  `./configure` invocation with `CC=cc GPUCXX=hipcc GPU_ARCH=gfx90a
  ROCM_PATH=$ROCM_PATH --use-hip=yes --use-mpi=yes --use-rccl=no
  --use-cudss=no` and `--lapack-lib-name=sci_amd`.
- `gkeyll-agent-dev/machines/mkdeps.frontier.sh` — builds SuperLU 7.0.0 and
  LuaJIT (git) under `$HOME/gkylsoft/`. Already run; deps installed.

### Files modified

- `core/zero/gkyl_util.h` — §3a-bis gate fix: `#ifdef __NVCC__` →
  `#if defined(GKYL_HAVE_GPU)`. Inline `#define GKYL_HAVE_CUDA` removed (now
  CFLAGS-driven). Inline `<cuda_runtime.h>` include replaced with the shim.
- `core/zero/gkyl_array.h`, `gkyl_alloc.h`, `gkyl_nccl_comm.h`,
  `gkyl_nccl_comm_priv.h` — added `#include <gkyl_gpu_runtime.h>` at top.
- `configure` — new flags: `GPUCXX`, `GPU_ARCH`, `ROCM_PATH`, `--use-hip`,
  `--use-rccl`, `--rccl-inc`, `--rccl-lib`. Mutual-exclusion validation for
  `--use-nccl`/`--use-rccl`. New keys propagated to `config.mak`.
- `Makefile` (top level) — `USING_HIPCC` block paralleling `USING_NVCC`.
  Unified `GPU_FLAGS` / `GPU_LDFLAGS` / `GPU_LIBS`. CFLAGS injection of
  `-DGKYL_HAVE_HIP -DGKYL_HAVE_GPU -D__HIP_PLATFORM_AMD__=1` on the HIP path
  (and `-DGKYL_HAVE_CUDA -DGKYL_HAVE_GPU` on the CUDA path).
  `BUILD_DIR=hip-build`. New RCCL gate. New exports (`USING_HIPCC GPU_FLAGS
  GPU_LDFLAGS GPU_LIBS ROCM_PATH USING_RCCL RCCL_INC_DIR RCCL_LIB_DIR
  RCCL_LIBS`). `GPUCXX` defaults to `$(CC)` on the NVCC path so the existing
  CUDA build keeps working.
- `Makefile_for_ext_C_input` — same `USING_HIPCC` and RCCL changes mirrored
  for downstream app builds (per plan §1e).
- `gkeyll/Makefile-gkeyll` — `${CUDA_LIBS}` → `${GPU_LIBS}` in the executable
  link line (per plan §1e).
- `core/Makefile-core` — `.cu` rule uses `$(GPUCXX) $(CFLAGS) $(GPU_FLAGS)`.
  RCCL include/lib added to `EXT_INCS`/`EXEC_LIB_DIRS`/`EXEC_EXT_LIBS`. The
  non-NVCC link branch switched from `${CUDA_LIBS}` to `${GPU_LIBS}` (so HIP
  builds pull `-lamdhip64`) plus `${RCCL_LIBS}` and `-L${RCCL_LIB_DIR}`. The
  `.cu` SRCS gate is intentionally still `USING_NVCC` only — Phase 2 widens
  this.

### Plan deviation worth folding into `amd_port_plan.md` §1c

Host CFLAGS on the HIP path **must** include `-D__HIP_PLATFORM_AMD__=1`. The
plan as written sets `-DGKYL_HAVE_HIP -DGKYL_HAVE_GPU -I$(ROCM_PATH)/include`
on host CFLAGS but that's not enough — `<hip/hip_runtime.h>` errors at line
66 of `hip_vector_types.h` without `__HIP_PLATFORM_AMD__` because hipcc
auto-injects it but Cray `cc` doesn't. AMD's `host_defines.h` then makes
`__device__` / `__host__` empty for non-hipcc TUs, so `GKYL_CU_DH`-annotated
`.c` kernel files compile cleanly under host `cc`. The Makefile comment
documents this; the plan should mirror it.

### How to rebuild from scratch

```bash
cd /autofs/nccsopen-svm1_home/junoravin/working_dev/gkeyll-agent-dev
source machines/configure.frontier-gpu.sh   # writes config.mak
rm -rf hip-build
make core -j 8
```

---

## Phase 2 — Next

**Goal:** `alloc.c` + `array.c` + the ~20 `_cu.cu` files in `core/zero/`
compile under hipcc and the resulting libg0core.so passes the first two
unit-test gates.

**Gate:** `ctest_alloc_cu` and `ctest_array_cu` pass on a compute node.

### Work items (per plan §3b)

1. **`core/zero/alloc.c`** — change `#include <cuda_runtime.h>` to
   `#include <gkyl_gpu_runtime.h>`. Flip outer `#ifdef GKYL_HAVE_CUDA` →
   `#ifdef GKYL_HAVE_GPU`. Body works as-is via the shim (cudaMalloc /
   cudaFree / etc. are aliased).
2. **`core/zero/array.c`** — same treatment. The `cudaStreamDestroy` site at
   `array.c:59` is the one cited in the plan.
3. **`core/zero/*_cu.cu`** files (~20) — change
   `#include <cuda_runtime.h>` → `#include <gkyl_gpu_runtime.h>`. Should
   be a single find/replace across these files. Source list:
   `array_ops_cu.cu`, `array_average_cu.cu`, `array_dg_reduce_cu.cu` (also
   touched in Phase 3), `array_integrate_cu.cu` (also Phase 3),
   `array_reduce_cu.cu` (also Phase 3), `cart_modal_*_cu.cu` (4 files),
   `dg_array_mask_cu.cu`, `dg_basis_ops_cu.cu`, `dg_bin_ops_cu.cu`,
   `dg_geom_cu.cu`, `dg_interpolate_cu.cu`, `nodal_ops_cu.cu`,
   `skin_surf_from_ghost_cu.cu`, `struct_of_arrays_cu.cu`,
   `tensor_field_ops_cu.cu`, plus any I missed — confirm with
   `find core/zero -name '*_cu.cu'`.
4. **Build-system flip**: in `core/Makefile-core`, widen the `.cu` SRCS
   gate from `ifdef USING_NVCC` to `ifdef USING_NVCC` ∨ `ifdef USING_HIPCC`.
   Same for the kernel-overrides block at `core/Makefile-core:90-114` and
   the `UNIT_CU_SRCS` block at lines 26-32 (but exclude `ctest_cusolver.cu`
   and `ctest_cudss.cu` — those stay CUDA-only per plan §7).
5. **Sanity check**: `make core -j 8` should now compile `.cu` files with
   hipcc. Expect new `*.cu.o` objects under `hip-build/core/zero/`.
6. **Build the unit tests**: `make core-unit -j 8`.
7. **Run the gate tests** on a compute node:
   ```bash
   salloc -A fus183 -p batch -N 1 -t 1:00:00 --gpus=1
   # inside the allocation:
   srun -A fus183 -n 1 -c 8 --gpus-per-task=1 \
       hip-build/core/unit/ctest_alloc_cu
   srun -A fus183 -n 1 -c 8 --gpus-per-task=1 \
       hip-build/core/unit/ctest_array_cu
   ```

### Likely fall-out to handle in Phase 2

- **`-fgpu-rdc` link**: hipcc relocatable device code is fragile if any TU
  is built without it. The current `core/Makefile-core` non-NVCC link branch
  uses `$(CC)` (Cray cc) — that's correct for host link, but device-symbol
  resolution from `.cu.o` files requires a hipcc-driven device link step.
  When .cu files start producing objects, the link line probably needs to
  go through `$(GPUCXX) --hip-link -fgpu-rdc` for the device side, or the
  Cray cc link needs `-Wl,--whole-archive` plus `$(GPU_LDFLAGS)`. cholla's
  `make.host.frontier` is the reference. Will surface as undefined-symbol
  errors at link time — not silent.
- **`hipHostMalloc` flag semantics**: plan §11 risk #4. Verify pinned-host
  I/O streams behave correctly. `ctest_alloc_cu` should catch it.
- **Kernel-launch syntax**: `kernel<<<grid, block>>>(args)` works under
  hipcc unmodified per plan §3b — no source changes expected. If it
  breaks, that's a hipcc version issue worth investigating.

### Files NOT to touch in Phase 2

- Reduction kernels (`array_reduce_cu.cu`, `array_dg_reduce_cu.cu`,
  `array_integrate_cu.cu`) need an additional include change beyond the
  shim — that's Phase 3 (`core/zero/gkyl_gpu_reduce.h` for hipCUB).
  Including only the shim per Phase 2 should still let them compile (CUB
  isn't pulled in via the shim path), but they'll fail at template
  instantiation. If the build fails on these three specifically, defer
  them by gating their `.cu` entries behind `ifndef USING_HIPCC` until
  Phase 3, or accept the failures and skip to Phase 3.
- `mat.c` (cuBLAS) — Phase 5.
- `nccl_comm.c` and friends — Phase 6.
- `cusolver_ops.cu`, `cudss_ops.cu` — out of scope (plan §7).

### Where to find things

- Shim header: [`gkeyll-agent-dev/core/zero/gkyl_gpu_runtime.h`](gkeyll-agent-dev/core/zero/gkyl_gpu_runtime.h)
- Top-level Makefile USING_HIPCC block: [`gkeyll-agent-dev/Makefile`](gkeyll-agent-dev/Makefile) (lines ~74-125)
- Configure script: [`gkeyll-agent-dev/configure`](gkeyll-agent-dev/configure)
- Frontier configure: [`gkeyll-agent-dev/machines/configure.frontier-gpu.sh`](gkeyll-agent-dev/machines/configure.frontier-gpu.sh)
- Frontier deps: [`gkeyll-agent-dev/machines/mkdeps.frontier.sh`](gkeyll-agent-dev/machines/mkdeps.frontier.sh)
- cholla reference (Frontier HIP recipe): [`cholla/builds/make.host.frontier`](cholla/builds/make.host.frontier), [`cholla/builds/setup.frontier.cce.sh`](cholla/builds/setup.frontier.cce.sh)
