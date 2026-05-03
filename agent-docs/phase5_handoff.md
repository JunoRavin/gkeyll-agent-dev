# AMD Port — Phase 5 Handoff

This file captures Phase 5 completion state and the Phase 6 plan, so a fresh
Claude session can pick up cleanly.

Source plan: [`amd_port_plan.md`](amd_port_plan.md). Earlier handoffs:
[`phase1_handoff.md`](phase1_handoff.md), [`phase2_handoff.md`](phase2_handoff.md),
[`phase3_handoff.md`](phase3_handoff.md), [`phase4_handoff.md`](phase4_handoff.md).
Read plan §6a (LU dependency), §8 (unit tests), and §10 (phasing) before
starting Phase 6.

---

## Environment

`PrgEnv-amd` + `rocm/6.2.4` + `craype-accel-amd-gfx90a` + `cray-mpich`
+ `cray-libsci`. Working tree
`/autofs/nccsopen-svm1_home/junoravin/working_dev/gkeyll-agent-dev/`. Account
`fus183`. Every Bash invocation prefixes the module-load +
`LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH`.

> **Note — `MPICH_GPU_SUPPORT_ENABLED=1` dropped at end of Phase 5.** Earlier
> handoffs prefixed every srun with this export. We verified in Phase 5 by
> running the full single-process `ctest_*` suite both with and without the
> flag — every test, including the test_3x_gpu p=2 carve-out, produces
> identical results. The flag only matters when MPICH needs to recognize a
> raw device pointer in `MPI_Send`/`MPI_Recv` (i.e. multi-process
> `mctest_mpi_comm` and the GPU-aware MPI path in `gkyl_mpi_comm.c`).
> Phase 6's RCCL work goes through RCCL's own ROCm transport, not MPICH —
> so the flag is unlikely to be needed there either, but verify before
> dropping it from the `mctest_nccl_comm` recipe in Phase 6.

rocBLAS lives at `/opt/rocm-6.2.4/include/rocblas/rocblas.h`; rocSOLVER at
`/opt/rocm-6.2.4/include/rocsolver/rocsolver.h`. Both ship as C-clean
(extern "C"-guarded) headers, so they include directly from C TUs without
the `extern "C++"` defense `gkyl_gpu_runtime.h` needs.

---

## Phase 5 — Complete

**Headline:** [`mat.c`](gkeyll-agent-dev/core/zero/mat.c) ports cleanly from
cuBLAS to rocBLAS + rocSOLVER. Both Phase 5 gate tests (`ctest_mat`) and the
Phase 4 LU-blocked carve-outs (`ctest_dg_bin_ops`, `ctest_array_average`)
flip from blocked / failing to green — except for one residual 3D+p2 path
in `ctest_array_average` that traces to an unrelated `gkyl_array_integrate`
bug exposed only after the LU stub asserted false (see "Carve-outs" below).

```text
hip-build/core/libg0core.so   25.7 MB → 25.9 MB
  17 *.cu.o objects in zero/  (unchanged inventory; mat.c flips its gate)
   6 *.cu.o objects in unit/  (unchanged from Phase 4)
```

### Gate test results (AMD MI250X, gfx90a, ROCm 6.2.4)

| Test | Pre-Phase-5 | Post-Phase-5 | Notes |
|---|---|---|---|
| `ctest_mat` | (Phase-5-blocked) | SUCCESS | Phase 5 primary gate. |
| `ctest_dg_bin_ops` | 15/24 (LU paths SIGABRT) | 24/24 | Was blocked on `gkyl_nmat_linsolve_lu_pa`; passes once mat.c is ported. |
| `ctest_array_average` | 3/6 (LU stub assert) | 5/6 | Two of three GPU paths flip to green; remaining `test_3x_gpu` p=2 fails in `gkyl_array_integrate_new` (unrelated to mat.c — see Carve-outs). |
| `ctest_array_reduce` | SUCCESS | SUCCESS | Re-verified — Phase 3 hipCUB path unaffected. |
| `ctest_array_dg_reduce` | SUCCESS | SUCCESS | Re-verified. |
| `ctest_array_integrate` | SUCCESS | SUCCESS | Re-verified — the only 3x-p2 failure surfaces via array_average, not array_integrate's own tests (its 3D paths are commented out per current state of the file). |
| `ctest_array_ops` | SUCCESS | SUCCESS | |
| `ctest_dg_array_mask` | SUCCESS | SUCCESS | |
| `ctest_struct_of_arrays` | SUCCESS | SUCCESS | |
| `ctest_tensor_field_ops` | SUCCESS | SUCCESS | |
| `ctest_basis` | SUCCESS | SUCCESS | |
| `ctest_dg_basis_ops` | SUCCESS | SUCCESS | |
| `ctest_skin_surf_from_ghost` | SUCCESS | SUCCESS | |

### Files created

- [`core/zero/gkyl_gpu_blas.h`](gkeyll-agent-dev/core/zero/gkyl_gpu_blas.h) — translation shim per plan §10 row 5. Single include point in `mat.c` and `gkyl_mat_priv.h`. Mirrors the Phase 1 (`gkyl_gpu_runtime.h`) and Phase 3 (`gkyl_gpu_reduce.h`) shim pattern. Direct symbol renames cover handle create/destroy, dgemm, dgemm_strided_batched. Static-inline wrappers handle batched LU (rocSOLVER's API differs from cuBLAS — takes `(m, n)` instead of `(n)` for dgetrf, uses strided pivots and omits the per-batch info pointer in dgetrs). A `gkyl_mat_to_blas_op()` helper translates `gkyl_mat_trans` to the platform's BLAS operation enum — needed because `rocblas_operation` uses values 111/112/113 while `cublasOperation_t` happens to coincide with `gkyl_mat_trans` (0/1/2). Pre-port `mat.c` was relying on the implicit cast working; the helper makes it portable.

### Files modified

- [`core/zero/mat.c`](gkeyll-agent-dev/core/zero/mat.c) — replaced `#include <cuda_runtime.h>` + `<cublas_v2.h>` with `#include <gkyl_gpu_blas.h>`; swept `GKYL_HAVE_CUDA → GKYL_HAVE_GPU` (14 sites); routed all three call sites that previously cast `gkyl_mat_trans` directly into a `cublasOperation_t` parameter through `gkyl_mat_to_blas_op()`. The `cublasDgetrfBatched`/`cublasDgetrsBatched` calls are unchanged at the call site — the shim's wrapper macros redirect them to the rocSOLVER-translated form on HIP.
- [`core/zero/gkyl_mat_priv.h`](gkeyll-agent-dev/core/zero/gkyl_mat_priv.h) — replaced `#include <cublas_v2.h>` with `#include <gkyl_gpu_blas.h>`; `GKYL_HAVE_CUDA → GKYL_HAVE_GPU` (2 sites).
- [`core/zero/gkyl_gpu_runtime.h`](gkeyll-agent-dev/core/zero/gkyl_gpu_runtime.h) — added the `__CUDA_ARCH__ ↔ __HIP_DEVICE_COMPILE__` alias (see "Bugs found and fixed" below). The HIP arm now injects `#define __CUDA_ARCH__ 1` during the device-compile pass so existing `#ifdef __CUDA_ARCH__` device gates in kernel files fire correctly. Long-form note in the session memory: `cuda_arch_macro_aliasing_under_hip.md`.
- [`Makefile`](gkeyll-agent-dev/Makefile) (top-level, USING_HIPCC arm) — `GPU_LIBS += -lrocblas -lrocsolver`. Mirrors the CUDA arm's `-lcublas -lcusparse -lcusolver`. `-lrocsparse` is not needed for the current `mat.c` surface (only batched LU + gemm) but should be added if a future kernel pulls it in.

### Bugs found and fixed (Phase-5-revealed, not Phase-5-introduced)

1. **`__CUDA_ARCH__` not defined under hipcc device pass.** Kernels in `core/ker/` (and a handful in `core/zero/*.c`) gate atomic-add accumulators on `#ifdef __CUDA_ARCH__`. nvcc sets that during device compile; hipcc instead sets `__HIP_DEVICE_COMPILE__`. Without translation, every kernel's device pass falls to the host branch under HIP and silently swaps `atomicAdd(&out[k], v)` for `out[k] += v` — racy across threads, silent wrong-answer bug. Surfaced when `ctest_array_average` started running after the Phase 5 mat.c port unblocked it. Fix: `gkyl_gpu_runtime.h` now `#define`s `__CUDA_ARCH__` to `1` whenever `__HIP_DEVICE_COMPILE__` is set, so the existing `#ifdef` fires correctly. ~99 affected sites across `core/ker/` and `core/zero/`; no per-site edits needed because the alias is transitive through `gkyl_util.h`. Long-form note saved in memory (`cuda_arch_macro_aliasing_under_hip.md`).

### Carve-outs documented (deferred bugs surfaced by Phase 5)

`ctest_array_average` — 1 of 6 tests still fails: `test_3x_gpu` for `poly_order=2`. Pinpointed via bisection to `gkyl_array_integrate_new(grid, basis, 1, OP_NONE, true)` for `ndim=3, poly_order=2`. The fault is a GPU memory access at a host-stack-like address (`0x7ffec5e00000`) — same family as the Phase 4 bug-class but in a different code path. **Not a regression introduced by Phase 5.** This is a latent bug in `gkyl_array_integrate`'s 3D+p2 GPU path that was hidden because:
- `ctest_array_integrate` itself has its 3D test commented out (see lines 554-557 of `core/unit/ctest_array_integrate.c`), so the 3D path was never directly exercised.
- `ctest_array_average` uses 3D integrate via its `solution_array_integrate` helper, but every GPU `test_*x_gpu` was previously blocked by the LU stub assert in `gkyl_nmat_cu_dev_new`, so the helper was never reached on AMD.

Phase 5 lifting both blockers (LU stub asserted false; `__CUDA_ARCH__` alias) is what surfaces it. **Recommended treatment:** track as a follow-up; it's the kind of bug the deferred §8c smoke test for `dg_geom`/`dg_interpolate`/`nodal_ops` would have caught, and it should be addressed alongside the simulation-app shakedown phase.

### How to rebuild + run from scratch

```bash
cd /autofs/nccsopen-svm1_home/junoravin/working_dev/gkeyll-agent-dev
module load PrgEnv-amd rocm/6.2.4 craype-accel-amd-gfx90a cray-mpich cray-libsci
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

rm -rf hip-build
make core -j 8           # <-- IMPORTANT: do not parallelize core + core-unit
make core-unit -j 8      #     ('make core core-unit -j N' has a libg0core.so race)

# Phase 5 gate tests:
for t in ctest_mat ctest_dg_bin_ops ctest_array_average; do
  srun -A fus183 -p batch -N 1 -n 1 -c 8 --gpus-per-task=1 -t 5:00 \
      hip-build/core/unit/$t
done

# Expected: ctest_mat and ctest_dg_bin_ops fully pass; ctest_array_average is
# 5/6 (the test_3x_gpu carve-out — see above).
```

---

## Phase 6 — Next

**Goal:** RCCL via macro switch — confirm `mctest_nccl_comm` passes single-node
under HIP. Per plan §10 row 6.

`gkyl_nccl_comm.h` and `gkyl_nccl_comm_priv.h` already had the
`#if defined(GKYL_HAVE_NCCL) || defined(GKYL_HAVE_RCCL)` gate added in Phase
1 per `phase1_handoff.md`, and the runtime shim include is also already in
place. The remaining work is largely a configure-time toggle plus a build
verification.

### Work items for Phase 6

1. **Add RCCL detect / link flags in `configure` and `Makefile`.** Mirror the
   existing NCCL paths. RCCL ships as `librccl.so` under `$ROCM_PATH/lib`
   (verify with `ls /opt/rocm-6.2.4/lib/librccl*`). Header at
   `<rccl/rccl.h>` (or possibly just `<rccl.h>`) — confirm before committing.

2. **Sweep `nccl_comm.c` and `multib_comm_conn_nccl.c`** so the same source
   compiles under both NCCL (CUDA) and RCCL (HIP). The two libraries are
   API-compatible by design (RCCL is a drop-in for NCCL); a thin
   `gkyl_gpu_collective.h` shim or simple symbol renames in the existing
   priv header should suffice.

3. **Decide `MPICH_GPU_SUPPORT_ENABLED` policy for `mctest_nccl_comm`.**
   The flag was dropped from the single-process unit-test recipe at end of
   Phase 5 after a side-by-side experiment showed zero impact (see the
   Environment note above). RCCL handles its GPU transports directly through
   ROCm — does NOT route through MPICH — so `mctest_nccl_comm` should also
   not need it. Verify by running the test both ways once it builds.
   `gkyl_mpi_comm.c`'s GPU-aware path (plan §11 risk #5) is a separate
   concern; it would need the flag if we exercise device pointers across
   `MPI_Send`/`MPI_Recv` but no current test does.

4. **Run `mctest_nccl_comm` single-node, 2 ranks.** Per plan §9 build recipe.

### Files NOT to touch in Phase 6

- `core/zero/cusolver_ops.cu`, `core/zero/cudss_ops.cu` — out of scope
  per plan §7 (CUDA-only linear solvers).
- `core/unit/ctest_cusolver.cu`, `core/unit/ctest_cudss.cu` — CUDA-only,
  excluded via `HIP_DEFER_UNIT` in `core/Makefile-core`.

### Where to find things

- Phase 1 runtime shim: [`gkeyll-agent-dev/core/zero/gkyl_gpu_runtime.h`](gkeyll-agent-dev/core/zero/gkyl_gpu_runtime.h)
- Phase 3 reduction shim: [`gkeyll-agent-dev/core/zero/gkyl_gpu_reduce.h`](gkeyll-agent-dev/core/zero/gkyl_gpu_reduce.h)
- Phase 5 BLAS shim: [`gkeyll-agent-dev/core/zero/gkyl_gpu_blas.h`](gkeyll-agent-dev/core/zero/gkyl_gpu_blas.h)
- Top-level Makefile USING_HIPCC block: [`gkeyll-agent-dev/Makefile`](gkeyll-agent-dev/Makefile)
- Sub-make: [`gkeyll-agent-dev/core/Makefile-core`](gkeyll-agent-dev/core/Makefile-core)
- Earlier handoffs: [`phase1_handoff.md`](phase1_handoff.md),
  [`phase2_handoff.md`](phase2_handoff.md), [`phase3_handoff.md`](phase3_handoff.md),
  [`phase4_handoff.md`](phase4_handoff.md)
- Long-form bug notes for AMD-port-relevant pitfalls live in this session's
  Claude memory dir (`~/.claude/projects/.../memory/`):
  - `array_reduce_gpu_output_pointer.md` (Phase 4)
  - `cu_dev_new_struct_inner_pointer_fixup.md` (Phase 4)
  - `cuda_arch_macro_aliasing_under_hip.md` (Phase 5)

### Deferred from Phase 5 (track separately)

- **`ctest_array_average` test_3x_gpu p=2** — fault in `gkyl_array_integrate_new`
  for ndim=3, poly_order=2. Pinpointed via bisection but not root-caused; not
  in mat.c. See "Carve-outs" above.
- **§8c smoke tests for `dg_geom`, `dg_interpolate`, `nodal_ops`,
  hybrid/gkhybrid bases** (Phase 4 carry-over). Compile-side green; runtime
  smoke coverage still pending. The 3D-p2 array_integrate carve-out above
  is the kind of bug these smoke tests would catch — adding them adjacent
  to the simulation-app shakedown phase remains the recommended approach.
- **Audit of remaining `GKYL_CU_D static ... choose_kernel` host-only
  dispatcher patterns.** Per-phase grep discipline applies; nothing
  surfaced in `mat.c` during Phase 5.
