# AMD Port — Vlasov V0 + V1 Handoff

Continuation of the AMD HIP/ROCm port effort. Core port closed at end of
Phase 6 (see [`phase6_handoff.md`](phase6_handoff.md)) with the §7 audit done
afterward. The Vlasov-up-the-stack effort is governed by
[`amd_vlasov_port_plan.md`](amd_vlasov_port_plan.md). This handoff captures
**V0 (diagnostic) and V1 (Moments build) complete**, paused before V2
(Vlasov library build).

## Environment

Unchanged from the core port. `PrgEnv-amd` + `rocm/6.2.4` +
`craype-accel-amd-gfx90a` + `cray-mpich` + `cray-libsci`. Working tree
`/autofs/nccsopen-svm1_home/junoravin/working_dev/gkeyll-agent-dev/`.
Account `fus183`. `MPICH_GPU_SUPPORT_ENABLED=1` is NOT set (Phase 5 found
it unnecessary for the unit-test path; will reassess for multi-GPU sim
runs at V6 if a fault surfaces).

---

## V0 — Done (diagnostic)

Flipped `machines/configure.frontier-gpu.sh` from `--app=core` to
`--app=vlasov`, sourced it, ran `make moments`. As predicted by Risk #3
in the plan, the build hit `-fgpu-rdc` link errors:

```
ld.lld: error: undefined hidden symbol: __hip_gpubin_handle_ce298529c58d2a85
>>> referenced by skin_surf_from_ghost_cu.cu
[+ four more __hip_gpubin_handle_* symbols, one per moments _cu.cu file]
make[1]: *** [Makefile-moments:105: ../hip-build/moments/libg0moments.so] Error 1
```

Diagnosis: `moments/Makefile-moments` had no `USING_HIPCC` arm in the lib
link rule, so the build fell through to `cc -shared` which can't perform
the relocatable-device-code link step that resolves these symbols. Identical
in structure to the bug Phase 2 fixed for `core/Makefile-core`.

V0 is diagnostic only — no remediation code in this phase.

---

## V1 — Done (Moments builds clean)

**Gate met:** `libg0moments.so` builds cleanly under HIP. `ldd` confirms
linkage against `librccl.so.1`, `libamdhip64.so.6`, `librocblas.so.4`,
`librocsolver.so.0`. Library size ~34 MB.

### Files modified in V1

```
machines/configure.frontier-gpu.sh    --app=core → --app=vlasov  (V0)
moments/Makefile-moments              5 edits (see below)
moments/zero/wave_geom.c              GKYL_HAVE_CUDA → GKYL_HAVE_GPU
moments/zero/wv_euler.c               same
moments/zero/wv_maxwell.c             same
moments/zero/wv_ten_moment.c          same
moments/unit/ctest_wave_geom.c        same
moments/unit/ctest_wv_euler.c         same
moments/unit/ctest_wv_maxwell.c       same
moments/unit/ctest_wv_ten_moment.c    same
```

**Deliberately NOT swept:** `moments/zero/fem_poisson.c`,
`moments/zero/gkyl_fem_poisson_priv.h`, `moments/unit/ctest_fem_poisson.c`,
`moments/unit/ctest_fem_poisson_vareps.c`, `moments/unit/ctest_fem_helmholtz.c`.
These stay on `GKYL_HAVE_CUDA` because the FEM Poisson GPU path reaches
into `gkyl_culinsolver_*` (cuSolver/cuDSS sparse LU) which is CUDA-only
per `amd_port_plan.md` §7 and the Phase 7 source-level gates added at
end of core port. See "FEM Poisson policy under HIP" below.

### Makefile-moments edits (mirror of core/Makefile-core Phase 2)

1. **`SRCS .cu` find: added `USING_HIPCC` arm with HIP_DEFER_SRCS filter**
   ```makefile
   ifdef USING_HIPCC
       HIP_DEFER_SRCS := zero/fem_poisson_cu.cu
       SRCS += $(filter-out $(HIP_DEFER_SRCS), \
                 $(shell find $(SRC_DIRS) -name *.cu))
       SRCS += $(shell find $(UNIT_DIRS) -name *.cu)
   endif
   ```
   `fem_poisson_cu.cu` calls `gkyl_culinsolver_*` directly (cuSolver-coupled);
   excluded under HIP same way `core/zero/cusolver_ops.cu` was. The four
   wave-equation `_cu.cu` files (`wv_euler_cu.cu`, `wv_maxwell_cu.cu`,
   `wv_ten_moment_cu.cu`, `wave_geom_cu.cu`) compile fine under hipcc.

2. **`UNIT_CU_SRCS`: added `USING_HIPCC` arm** with the same source list
   as NVCC (4 wave-equation `_cu.cu` tests).

3. **`EXT_INCS` / `EXEC_LIB_DIRS` / `EXEC_EXT_LIBS`:** added `RCCL_INC_DIR`,
   `RCCL_LIB_DIR`, `RCCL_LIBS`; replaced `${CUDA_LIBS}` with `${GPU_LIBS}`
   (the unified BUILD_APP-aware variable that resolves to cuBLAS+cuSolver
   on CUDA, rocBLAS+rocSOLVER on HIP).

4. **`.cu.o` build rule:** changed `$(CC) ... $(NVCC_FLAGS)` →
   `$(GPUCXX) ... $(GPU_FLAGS)`. Without this, the build invokes Cray `cc`
   on `.cu` files, which falls through to clang's CUDA mode and fails with
   "cannot find libdevice for sm_35".

5. **Kernel-override block: kept gate at `USING_NVCC` only** (NOT widened
   to `USING_NVCC ∨ USING_HIPCC` like core's). Under HIP `fem_poisson_cu.cu`
   is excluded so no device code calls the fem_poisson kernels — they only
   need to compile as plain host C through normal `cc`. Compiling them
   with `-x hip` surfaced a header-vs-`.c` `GKYL_CU_DH` annotation
   mismatch in `fem_poisson_bias_plane_lhs.c` (header declares with
   `GKYL_CU_DH`, definition omits it — see "Known issue" below).

6. **Lib link rule: added `else ifdef USING_HIPCC` arm** that drives the
   link through `$(GPUCXX) -shared $(GPU_LDFLAGS) -fPIC` (`hipcc
   --hip-link -fgpu-rdc`). Resolves the `__hip_gpubin_handle_*` symbols.

### FEM Poisson policy under HIP

The fem_poisson GPU path is fully CUDA-only on AMD because it depends on
`gkyl_culinsolver_*` (the sparse LU wrapper) which has no HIP backend in
this port. Under HIP:

- `fem_poisson_cu.cu` is filter-out'd from compilation (HIP_DEFER_SRCS).
- `fem_poisson.c`'s `#ifdef GKYL_HAVE_CUDA if (use_gpu) ... else ...`
  blocks fall through to the `#else` branch (SuperLU host LU).
- A user calling `gkyl_fem_poisson_new(use_gpu=true)` under HIP gets a
  silently host-side fem_poisson — no abort, no error, but the rest of
  their code expecting GPU-side data may not get what it wants.

**Implication for sim coverage:** Vlasov–Maxwell sims (the V4/V5 fiducials)
do NOT use FEM Poisson — they evolve EM fields hyperbolically. Vlasov–
Poisson sims (`rt_vp_*`) DO use FEM Poisson and will not run on AMD with
useful GPU acceleration until a HIP sparse-solver port (rocSPARSE-based)
lands. Out of scope for this round.

### Known issue (sidestepped, not fixed)

`moments/ker/fem_poisson/fem_poisson_bias_plane_lhs.c` has function
definitions that lack the `GKYL_CU_DH` annotation that the corresponding
header declarations carry:

```c
// Header (gkyl_fem_poisson_kernels.h):
GKYL_CU_DH void fem_poisson_bias_plane_lhs_1x_ser_p1_inx(...);

// Definition (fem_poisson_bias_plane_lhs.c):
void fem_poisson_bias_plane_lhs_1x_ser_p1_inx(...) { ... }
```

Under regular `cc` (where `GKYL_CU_DH` expands to nothing) this is
harmless. Under hipcc with `-x hip` it produces an
`__host__ function ... cannot overload __host__ __device__ function`
error, because the macro expands to the device annotation in the
header but not the definition.

V1 sidesteps this by NOT compiling the fem_poisson kernel files under
hipcc (kernel-override gated to `USING_NVCC` only). The file is a real
bug — adding `GKYL_CU_DH` to the definitions would fix it cleanly — but
since FEM Poisson is fully CUDA-only on AMD anyway, fixing this is
deferred to whichever future effort ports rocSPARSE.

---

## V2 — Next (Vlasov library build)

Same pattern as V1, applied to `vlasov/`:

1. **`vlasov/Makefile-vlasov`:** mirror the same six Makefile-moments edits.
   Vlasov has 45 `_cu.cu` files in `vlasov/zero/` (vs. 5 in moments) — expect
   one or more stack-frame-overflow link errors on the bigger
   poly_order × dim instantiations (Risk #2 in the plan); fix is the same
   three-option approach the core port used (LDS, `__constant__`, kernel
   split). Inventory of vlasov kernel directories that need the
   `-x hip` kernel-override widening will need to be enumerated by walking
   `vlasov/ker/*/`.

2. **Source-level `GKYL_HAVE_CUDA → GKYL_HAVE_GPU` sweep:** ~75 host files
   in `vlasov/zero/` per the plan's surface inventory. Watch for any that
   reach into `gkyl_culinsolver_*` (none expected, but verify with a
   grep before sweeping). The 11 unit tests in `vlasov/unit/` and 3 in
   `vlasov/creg/` carry gates too.

3. **Special case:**
   `vlasov/zero/vlasov_lte_proj_on_basis_cu.cu` includes `<cublas_v2.h>`
   directly. Replace with `#include <gkyl_gpu_blas.h>` and route any raw
   `gkyl_mat_trans` casts through `gkyl_mat_to_blas_op()`, mirroring the
   Phase 5 mat.c pattern.

4. **`vlasov/Makefile-vlasov` HIP_DEFER list:** check whether any vlasov
   `_cu.cu` file calls `gkyl_culinsolver_*`. If yes, add to HIP_DEFER (a
   grep will tell us — none expected for Vlasov–Maxwell paths).

V2 gate: `make vlasov` builds `libgvlasov.so` cleanly.

V3 work item then handles `vlasov/apps/` (`vlasov_comms.c` and
`vlasov_lw.c`) — both call `gkyl_nccl_comm_new`, gates need widening to
the `defined(NCCL) || defined(RCCL)` disjunction (~2 files).

V4/V5 are the single-GPU fiducial runs (`rt_vlasov_twostream_p2`,
`rt_vlasov_weibel_2x2v_p2`); V6 is multi-GPU.

### How to rebuild + run V0/V1 from scratch

```bash
cd /autofs/nccsopen-svm1_home/junoravin/working_dev/gkeyll-agent-dev
module load PrgEnv-amd rocm/6.2.4 craype-accel-amd-gfx90a cray-mpich cray-libsci
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

source machines/configure.frontier-gpu.sh   # has --app=vlasov as of V0
rm -rf hip-build
make moments -j 32                          # V1 gate

# Expected outcome:
#   * libg0moments.so in hip-build/moments/, ~34 MB
#   * fem_poisson_cu.cu skipped (filter-out'd from SRCS under HIP)
#   * Wave equation _cu.cu files compiled and present in
#     hip-build/moments/zero/wv_*_cu.cu.o + wave_geom_cu.cu.o
#   * Linker pulls in librccl.so.1, libamdhip64.so.6, librocblas.so.4,
#     librocsolver.so.0
```

### Where to find things

- This handoff: `/autofs/nccsopen-svm1_home/junoravin/working_dev/vlasov_v0_v1_handoff.md`
- Vlasov port plan: [`amd_vlasov_port_plan.md`](amd_vlasov_port_plan.md)
- Core port plan: [`amd_port_plan.md`](amd_port_plan.md)
- Earlier handoffs: [`phase1_handoff.md`](phase1_handoff.md) through
  [`phase6_handoff.md`](phase6_handoff.md)
- Long-form bug notes for AMD-port-relevant pitfalls in
  `~/.claude/projects/.../memory/`:
  - `array_reduce_gpu_output_pointer.md` (Phase 4)
  - `cu_dev_new_struct_inner_pointer_fixup.md` (Phase 4)
  - `cuda_arch_macro_aliasing_under_hip.md` (Phase 5)
  - `nccl_post_2_19_group_call_contract.md` (Phase 6)

### Open follow-ups (not blocking V2)

1. **`fem_poisson_bias_plane_lhs.c` annotation mismatch.** Real bug in
   the kernel file; sidestepped here by gating the kernel-override to
   NVCC-only. Adding `GKYL_CU_DH` to the function definitions would fix
   it. File this against the rocSPARSE-port follow-up; not on the
   Vlasov-Maxwell critical path.

2. **`fem_poisson_new(use_gpu=true)` silently degrading on AMD.** A user
   running a Vlasov-Poisson Lua script on AMD won't get an error — they'll
   get host-side SuperLU. Document this in the Vlasov-Poisson Lua
   wrapper or add a warning at sim startup.

3. **Moments unit tests on GPU (post-V6).** The plan deferred these.
   Per `amd_vlasov_port_plan.md` §"Unit Tests", revisit only if a fiducial
   sim surfaces a Moments correctness issue.
