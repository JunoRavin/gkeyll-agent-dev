# AMD Port — Vlasov + Moments Plan

Continuation of [`amd_port_plan.md`](amd_port_plan.md) (which covered the core
library). The core port closed with all 36 single-process GPU unit tests
passing on AMD MI250X (gfx90a) under ROCm 6.2.4 + RCCL 2.20.5 (see
[`phase6_handoff.md`](phase6_handoff.md), follow-up audit added Phase-7-style
linear-solver source-level gates). This plan extends the same approach up
the layer stack to Vlasov, with Moments as a build-time dependency.

End-state goal: run **single- and multi-GPU Vlasov–Maxwell** simulations on
Frontier's AMD MI250X. Two fiducials gate completion:
[`vlasov/luareg/rt_vlasov_twostream_p2.lua`](../gkeyll-agent-dev/vlasov/luareg/rt_vlasov_twostream_p2.lua)
and
[`vlasov/luareg/rt_vlasov_weibel_2x2v_p2.lua`](../gkeyll-agent-dev/vlasov/luareg/rt_vlasov_weibel_2x2v_p2.lua).

## Scope decisions

**In scope:**
- Full GPU coverage of `vlasov/zero/` and `vlasov/apps/` (45 `_cu.cu` files;
  ~75 host files with `GKYL_HAVE_CUDA` gates).
- Build coverage of `moments/zero/` (5 `_cu.cu` files; ~6 host files with
  gates) — Vlasov build-depends on Moments (`vlasov` target depends on
  `moments` in the top Makefile).
- Lua scripting (`vlasov/apps/vlasov_lw.c`) since the fiducial sims are
  Lua-driven.
- Fiducial single-GPU runs of the two named simulations.
- Single-node multi-GPU run of the same fiducials.

**Deferred:**
- **Moments unit-test correctness on GPU.** The user's prior on this is
  that `moments/unit/ctest_*` carry tight numerical tolerances that may
  spuriously fail on AMD without indicating a real bug. Moments needs to
  *build* in this plan; tolerance audit happens only if a Vlasov sim
  surfaces a Moments correctness issue.
- **Vlasov unit-test correctness on GPU** beyond what's needed to gate
  the build — tolerances in `vlasov/unit/` may need a similar audit, but
  the gating signal is the fiducial sim, not the unit tests.
- `gyrokinetic/`, `pkpm/`, `moments/amr/`, `moments/amr_creg/` —
  out of scope for this round.
- Linear solvers (`cusolver_ops.cu`, `cudss_ops.cu`) — already gated as
  CUDA-only per [`amd_port_plan.md`](amd_port_plan.md) §7.

## Reuse from the core port

**No new shims.** The three core shims already carry everything the
upper layers need:
- `core/zero/gkyl_gpu_runtime.h` — CUDA Runtime ↔ HIP runtime.
- `core/zero/gkyl_gpu_reduce.h` — CUB ↔ hipCUB.
- `core/zero/gkyl_gpu_blas.h` — cuBLAS ↔ rocBLAS+rocSOLVER (used by exactly
  one Vlasov file: `vlasov/zero/vlasov_lte_proj_on_basis_cu.cu`).

**Same macro scheme.** `GKYL_HAVE_CUDA` (CUDA only), `GKYL_HAVE_HIP` (HIP
only), `GKYL_HAVE_GPU` (either). The per-app macros `GKYL_HAVE_VLASOV` and
`GKYL_HAVE_MOMENTS` are orthogonal — they gate which subdirs build, not
which GPU runtime is in use.

**Same gate-sweep pattern.** Host `.c` files and priv headers whose `_cu.cu`
companions need to compile under HIP get their `#ifdef GKYL_HAVE_CUDA` gates
flipped to `GKYL_HAVE_GPU`. This is the Phase 2 finding from the core port
(`phase2_handoff.md` §5) — anything left on the narrow gate silently
no-ops the GPU branch under HIP, with no compile-time error.

## Build system

The configure side already supports `--app=vlasov`; the top Makefile already
handles `BUILD_APP=vlasov` by setting `HAVE_APP_FLAGS = -DGKYL_HAVE_VLASOV
-DGKYL_HAVE_MOMENTS` (Makefile lines 39-41). So the Phase 0 work is just to
flip [`machines/configure.frontier-gpu.sh`](../gkeyll-agent-dev/machines/configure.frontier-gpu.sh)
from `--app=core` to `--app=vlasov`. The build then exercises:
- `moments`, `moments-unit`, `moments-regression` make targets
- `vlasov`, `vlasov-unit`, `vlasov-regression` make targets

The `_cu.cu` find-glob in `core/Makefile-core` does not extend to
`moments/` or `vlasov/`. Each layer has its own `Makefile-moments` /
`Makefile-vlasov` that needs the same `USING_HIPCC` arm wiring as
`core/Makefile-core` (the kernel-override `-x hip` block, the `.cu` find,
the device-link recipe). This is the bulk of the build-system work in
this plan.

## Surface inventory

Counts via `grep -rl GKYL_HAVE_CUDA <dir>`:

| Area | `_cu.cu` files | Host `.c` files w/ `GKYL_HAVE_CUDA` gates |
|---|---:|---:|
| `moments/zero/` | 5 | 6 |
| `moments/unit/` | — | 7 |
| `vlasov/zero/` | 45 | 75 |
| `vlasov/apps/` | 0 | 2 |
| `vlasov/unit/` | — | 11 |
| `vlasov/creg/` | — | 3 |

The five Moments `_cu.cu` files are: `fem_poisson_cu.cu`, `wave_geom_cu.cu`,
`wv_euler_cu.cu`, `wv_maxwell_cu.cu`, `wv_ten_moment_cu.cu`. The user's
note that "some `.cu` files create `wv_eqn` objects on device" specifically
points at the three `wv_*_cu.cu` files.

## Special dependencies

Three call-out items where a Vlasov file reaches into core-port territory
and needs the corresponding shim to be already in place (it is):

1. **`vlasov/zero/vlasov_lte_proj_on_basis_cu.cu`** includes `<cublas_v2.h>`
   directly. Replace with `#include <gkyl_gpu_blas.h>` and route any
   operation-enum casts through `gkyl_mat_to_blas_op()` per the Phase 5
   pattern in `core/zero/mat.c`.

2. **`vlasov/apps/vlasov_comms.c` and `vlasov/apps/vlasov_lw.c`** call
   `gkyl_nccl_comm_new(...)`. The Phase 6 gate widening in
   `core/zero/gkyl_nccl_comm.h` already handles the include switch and
   the `defined(GKYL_HAVE_NCCL) || defined(GKYL_HAVE_RCCL)` disjunction
   for the symbols. Verify these two app files use the same disjunction
   (likely they currently gate on `GKYL_HAVE_NCCL` only — sweep them).

3. **Lua wrapper (`vlasov_lw.c`)** is the entry point the fiducial sims
   reach the Vlasov app through. Confirm that the LuaJIT path links
   cleanly under hipcc — Phase 1 of the core port set up LuaJIT linking
   in the unified Makefile, so this should be a check, not new work.

## Unit Tests

Per the user's deferral, the goal here is **build coverage**, not pass.
Three concrete steps:

1. **Confirm `moments-unit` and `vlasov-unit` build cleanly** under HIP
   once the gate sweeps land. `make moments` and `make vlasov` (which
   build the libraries) is the gate; unit-test binaries can fail to
   *run* and we still proceed to fiducial sims.
2. **Run `moments-check` and `vlasov-check`** as a record-keeping pass —
   capture which tests pass / fail / segfault so we can revisit
   tolerances later. Don't tune tolerances now.
3. **Triage post-fiducial.** If `rt_vlasov_twostream_p2` runs and produces
   plausible output, the unit-test failures are either (a) tolerance
   issues (defer) or (b) latent kernel bugs that didn't surface in the
   fiducial (defer; track separately, like the
   `ctest_array_average` test_3x_gpu p=2 carve-out from the core port).

## Frontier Build & Test Recipe

```bash
cd /autofs/nccsopen-svm1_home/junoravin/working_dev/gkeyll-agent-dev
module load PrgEnv-amd rocm/6.2.4 craype-accel-amd-gfx90a cray-mpich cray-libsci
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

# Reconfigure to --app=vlasov (rebuilds moments transitively).
source machines/configure.frontier-gpu.sh   # already loads modules + ./configure

rm -rf hip-build
make moments -j 32      # builds libgmoments.so or equivalent
make vlasov  -j 32      # builds libgvlasov.so + the Vlasov app frontend

# Inside an allocation:
salloc -A fus183 -N 1 --gpus 1 -t 30:00

# Single-GPU fiducial:
srun -n 1 -c 8 --gpus-per-task=1 \
     hip-build/vlasov/luareg/rt_vlasov_twostream_p2

# Multi-GPU fiducial (single node, 2 GPUs):
srun -n 2 -c 8 --gpus-per-task=1 \
     hip-build/vlasov/luareg/rt_vlasov_weibel_2x2v_p2
```

## Phasing

Mirror the structure of `amd_port_plan.md` §10 but with the smaller scope
the upper layers warrant. **Phase numbering continues from the core port
where each phase number was a build-flag/library milestone.** No
gate-to-next-phase entries are blocking on linear solvers (out of scope per
core-port §7).

| Phase | Deliverable | Gate to next phase |
|---|---|---|
| **V0** | Reconfigure to `--app=vlasov`; attempt `make moments`; collect first compile errors | `make moments` reaches the link step (errors are content errors, not "configure didn't extend") |
| **V1** | Sweep `moments/zero/` `GKYL_HAVE_CUDA` → `GKYL_HAVE_GPU` (~6 host files + 7 unit tests); extend `moments/Makefile-moments` with the `USING_HIPCC` arm mirroring `core/Makefile-core` | `make moments` builds `libgmoments.so` cleanly |
| **V2** | Sweep `vlasov/zero/` (~75 host files); extend `vlasov/Makefile-vlasov` with the `USING_HIPCC` arm; route `vlasov_lte_proj_on_basis_cu.cu` through `gkyl_gpu_blas.h` | `make vlasov` builds `libgvlasov.so` cleanly |
| **V3** | Sweep `vlasov/apps/` (2 files: `vlasov_comms.c`, `vlasov_lw.c`) — NCCL gate widening + LuaJIT link verification | `make vlasov` produces the Vlasov-app driver binary, links against `librccl.so` |
| **V4** | Single-GPU run of `rt_vlasov_twostream_p2` to convergent output (no abort, plausible final state). The 1x1v simplest case should surface only the most glaring kernel bugs | sim completes |
| **V5** | Single-GPU run of `rt_vlasov_weibel_2x2v_p2` (2x2v phase space — exercises the harder kernel paths and 4-D field updates) | sim completes |
| **V6** | Multi-GPU (2 ranks, single node) run of one of the two fiducials | sim completes; output matches single-GPU within tolerance |

V0 is intentionally light — it's diagnostic, not deliverable. V4 and V5
are separated because the 1x1v case has a much narrower kernel surface
than 2x2v; surfacing 1x1v bugs first keeps the search space small.

## Risks and Open Questions

1. **Vlasov kernel-file `__CUDA_ARCH__` density.** The Phase 5 alias
   (`__CUDA_ARCH__=1` during the hipcc device pass, defined in
   `core/zero/gkyl_gpu_runtime.h`) protects against the silent atomic-add
   degradation across all of `vlasov/ker/` automatically. But the alias
   only fires if the kernel files include `gkyl_util.h` (transitively
   pulls in the runtime shim). Verify on a sample kernel file under
   `vlasov/ker/` before banking on it.

2. **Stack-frame size of large Vlasov device-side helpers.** `core` had
   one casualty (`ker_dev_cu_ser_2d` in `ctest_basis_cu.cu`, fixed by
   resizing local arrays). `vlasov/ker/` has many more poly_order × dim
   combinations; expect 1-3 stack-frame-overflow link errors on the V2
   gate. Three-option fix is the same as core: spill to LDS, hoist to
   `__constant__`, or split into smaller kernels.

3. **`-fgpu-rdc` link-step memory pressure.** `core` was at 17 + 6 = 23
   device TUs; vlasov/zero alone adds 45, and there are likely more in
   `vlasov/ker/`. Cholla hits memory limits around 100+ device TUs on
   Frontier — we may bump up against that on V2/V3. If so, the link
   command may need an explicit `-fno-gpu-rdc` for a few isolated TUs
   that don't actually have cross-TU device symbols, or the link may
   need to chunk via `--hip-link --offload-link`. Watch for OOM on the
   final `libgvlasov.so` link.

4. **`MPICH_GPU_SUPPORT_ENABLED` for multi-GPU.** Single-GPU runs go
   through RCCL collectives only and the flag is not needed (verified at
   end of Phase 5). Multi-GPU may exercise `gkyl_mpi_comm.c`'s GPU-aware
   path *if* the Vlasov app uses raw `MPI_Send`/`MPI_Recv` on device
   pointers. If V6 fails with weird intra-node errors, set
   `MPICH_GPU_SUPPORT_ENABLED=1` for that specific run.

5. **`MPICH_SMP_SINGLE_COPY_MODE=NONE` on the dev sub-system.** The
   `_host` bcast tests in `mctest_nccl_comm` needed this due to XPMEM
   permission restrictions. The Vlasov app may hit the same path during
   I/O bcast at sim startup. Pre-set on the dev sub-system; verify on
   production Frontier.

6. **Moments unit-test failures masking real Moments correctness bugs.**
   Per the deferral above, we tolerate Moments unit-test failures during
   V1-V3. If the V4 single-GPU sim produces clearly wrong physics
   (instead of just non-convergent or aborted), the suspect order is
   (a) Moments wave-equation kernels (`wv_euler_cu.cu`,
   `wv_maxwell_cu.cu`, `wv_ten_moment_cu.cu`) — re-enable their unit
   tests and tune tolerances at that point.

7. **`vlasov/apps/vlasov_lw.c` Lua-side calls into NCCL.** If the Lua
   regression infrastructure constructs the comm differently from the
   `mctest_nccl_comm` test we already validated, V3 may surface a
   different NCCL call sequence we haven't exercised. Most likely
   suspect: `gkyl_comm_array_bcast_host` for the initial-condition I/O.
