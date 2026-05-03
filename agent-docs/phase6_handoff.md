# AMD Port — Phase 6 Handoff

This file captures Phase 6 completion state, so a fresh Claude session can
pick up the post-port follow-ups cleanly. Phase 6 closes the AMD port effort
described in [`amd_port_plan.md`](amd_port_plan.md); there is no Phase 7
(linear-solver porting was deliberately scoped out per plan §7).

Earlier handoffs: [`phase1_handoff.md`](phase1_handoff.md),
[`phase2_handoff.md`](phase2_handoff.md), [`phase3_handoff.md`](phase3_handoff.md),
[`phase4_handoff.md`](phase4_handoff.md), [`phase5_handoff.md`](phase5_handoff.md).

---

## Environment

`PrgEnv-amd` + `rocm/6.2.4` + `craype-accel-amd-gfx90a` + `cray-mpich` +
`cray-libsci`. RCCL ships at `/opt/rocm-6.2.4/include/rccl/rccl.h` and
`/opt/rocm-6.2.4/lib/librccl.so` — version 2.20.5 (`NCCL_VERSION_CODE 22005`).

> **No `MPICH_GPU_SUPPORT_ENABLED=1` required.** Phase 5 dropped this flag
> from the unit-test recipe after a side-by-side experiment. Phase 6
> confirmed the same applies to `mctest_nccl_comm`: RCCL routes its GPU
> transports directly through ROCm and never goes through MPICH. The flag
> is only needed if `gkyl_mpi_comm.c`'s GPU-aware path (raw device pointers
> across `MPI_Send`/`MPI_Recv`) is exercised — no current test does.

> **`MPICH_SMP_SINGLE_COPY_MODE=NONE` needed for host-buffer broadcasts on
> this dev sub-system.** The two `nccl_bcast_*_host` tests use the
> underlying MPI host bcast which Cray MPICH defaults to XPMEM single-copy
> for intra-node transfer. XPMEM kernel module access is restricted on the
> dev sub-system (`xpmem_attach error: Permission denied`). Setting
> `MPICH_SMP_SINGLE_COPY_MODE=NONE` falls back to two-step copy through
> POSIX shm — slower but works. **This is environment-specific.** Production
> Frontier likely has XPMEM accessible and won't need the flag; verify on
> the actual run target.

---

## Phase 6 — Complete

**Headline:** RCCL is wired in via the existing `nccl_comm.c` source plus a
gate widening to recognize `GKYL_HAVE_RCCL`. All 18 NCCL tests pass with 2
ranks on AMD. The migration also fixed four pre-existing NCCL-API-contract
violations in the original code that surfaced as a SIGSEGV in
`nccl_n2_per_sync_2d` against RCCL 2.20.5.

```text
hip-build/core/libg0core.so   25.9 MB → 25.9 MB (no significant size change)
  17 *.cu.o objects in zero/  (unchanged inventory)
   6 *.cu.o objects in unit/  (unchanged from Phase 4)
mctest_nccl_comm: linked against /opt/rocm-6.2.4/lib/librccl.so.1 (verified
  via ldd)
```

### Gate test result (AMD MI250X, gfx90a, ROCm 6.2.4, RCCL 2.20.5)

| Test | Pre-Phase-6 | Post-Phase-6 |
|---|---|---|
| `mctest_nccl_comm` (2 ranks, `-M`) | did not link (RCCL not enabled) | **18/18 SUCCESS** on both ranks |

Run command:
```bash
MPICH_SMP_SINGLE_COPY_MODE=NONE \
  srun -A fus183 -p batch -N 1 -n 2 -c 8 --gpus-per-task=1 -t 5:00 \
       hip-build/core/unit/mctest_nccl_comm -M
```

(`-M` is acutest's MPI-init opt-in — required for any `mctest_*` since
acutest's default main does not call `MPI_Init`.)

### Files modified in Phase 6

#### Gate widening (`GKYL_HAVE_NCCL` → `defined(GKYL_HAVE_NCCL) || defined(GKYL_HAVE_RCCL)`)

- [`core/zero/gkyl_nccl_comm.h`](gkeyll-agent-dev/core/zero/gkyl_nccl_comm.h) — gate widened, plus a conditional include switch (`<rccl/rccl.h>` on the RCCL arm, `<nccl.h>` on the NCCL arm). The rest of the codebase keeps using the platform-agnostic `nccl*` API since RCCL is symbol-compatible (same enum values, same calling conventions).
- [`core/zero/nccl_comm.c`](gkeyll-agent-dev/core/zero/nccl_comm.c) — gate widened. Same single source compiles for both NCCL and RCCL; **no shim header was needed for the runtime API**.
- [`core/zero/multib_comm_conn_nccl.c`](gkeyll-agent-dev/core/zero/multib_comm_conn_nccl.c) — gate widened. Existing `#else` assert-stub fallback now triggers only when neither library is enabled.
- [`core/unit/mctest_nccl_comm.c`](gkeyll-agent-dev/core/unit/mctest_nccl_comm.c) — gate widened.

#### Configure flip

- [`machines/configure.frontier-gpu.sh`](gkeyll-agent-dev/machines/configure.frontier-gpu.sh) — `--use-rccl=no` → `--use-rccl=yes`, plus `--rccl-inc=$ROCM_PATH/include` and `--rccl-lib=$ROCM_PATH/lib` to match. The Phase 1 Makefile plumbing (`USE_RCCL`, `RCCL_LIBS=-lrccl`, `-DGKYL_HAVE_RCCL`, `-I/-L` paths, link line) was already in place — no Makefile edits needed.

#### NCCL API-contract fixes (independent of RCCL, applied in `nccl_comm.c`)

The user supplied a corrected `nccl_comm.c` with four fixes addressing API-contract violations that NCCL ≤2.18 silently tolerated but NCCL ≥2.19 (and RCCL 2.20.5) do not. All four landed in our tree on top of the gate widening:

1. **`array_per_sync`: consolidated per-direction NCCL groups into one** — peer ranks on opposite edges had previously been split across separate group calls, violating NCCL's requirement that matching send/recv ops fuse in a single group to make progress concurrently.

2. **`array_sync` and `array_per_sync`: separated buffer prep from NCCL calls** — now Phase 1 (range/buffer prep) is outside the group, Phase 2 (group with only `ncclSend`/`ncclRecv`) is inside, Phase 3 (`cudaStreamSynchronize`) is after, Phase 4 (`copy_from_buffer`) is after sync. Helper `recv_nids[]` / `send_nids[]` arrays carry neighbor IDs across phases. NCCL ≥2.19 may defer stream operations from inside a group; CUDA memcpy submitted inside the group can execute out of order with the deferred NCCL ops.

3. **`array_per_sync`: guarded the group on `has_nccl_ops`** — interior ranks in a `{4,1}` phase-space decomp `touches_any_edge` (the velocity dim) but have no periodic neighbors in any periodic config-space direction. The pre-fix code would post an empty `ncclGroupStart`/`ncclGroupEnd` pair, which segfaults in NCCL 2.25 (and matches the `nccl_n2_per_sync_2d` SIGSEGV we hit against RCCL 2.20.5). The guard skips the group entirely on ranks with nothing to post.

4. **`barrier`: added `MPI_Barrier(nccl->mcomm)`** — the previous implementation only called `cudaStreamSynchronize` (rank-local). Without a cross-rank barrier, interior ranks returning early from `array_per_sync` could race ahead and submit new NCCL group ops while edge ranks were still inside their per-sync group, violating NCCL's same-issuing-order-across-ranks requirement.

Long-form note saved in memory (`nccl_post_2_19_group_call_contract.md`); the user's reference notes preserved at [`NCCL_MIGRATION_NOTES.md`](NCCL_MIGRATION_NOTES.md) (peer to this handoff under `working_dev/`).

### How to rebuild + run from scratch

```bash
cd /autofs/nccsopen-svm1_home/junoravin/working_dev/gkeyll-agent-dev
module load PrgEnv-amd rocm/6.2.4 craype-accel-amd-gfx90a cray-mpich cray-libsci
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

source machines/configure.frontier-gpu.sh    # also runs ./configure with --use-rccl=yes
rm -rf hip-build
make core -j 8
make core-unit -j 8

# Phase 6 gate test (compute node, single node, 2 ranks):
MPICH_SMP_SINGLE_COPY_MODE=NONE \
  srun -A fus183 -p batch -N 1 -n 2 -c 8 --gpus-per-task=1 -t 5:00 \
       hip-build/core/unit/mctest_nccl_comm -M

# Expected: 18/18 SUCCESS on both ranks.
```

---

## Port complete — recommended follow-ups (no Phase 7)

The AMD HIP/ROCm port of the core library is functionally complete. Below
are the loose ends tracked across the port that didn't fit into a phase,
ordered by likely priority.

1. **`ctest_array_average` test_3x_gpu p=2 carve-out (Phase 5).**
   `gkyl_array_integrate_new` for `ndim=3, poly_order=2` faults on AMD.
   Pinpointed via bisection but not root-caused; not in `mat.c`. The bug
   was hidden because `ctest_array_integrate`'s 3D path is commented out
   (lines 554-557 of that file). Fix probably also resolves a future
   issue in 3D simulation apps.

2. **§8c smoke tests deferred from Phase 4.** `dg_geom_cu.cu`,
   `dg_interpolate_cu.cu`, `nodal_ops_cu.cu` all compile cleanly under
   hipcc (verified at end of Phase 4) but have no runtime smoke-test
   coverage. Two `ctest_basis_cu.cu` extensions for hybrid/gkhybrid bases
   also pending. Add adjacent to simulation-app shakedown — the
   array_average 3D-p2 carve-out above is exactly the kind of bug these
   tests would catch.

3. **`gkyl_mpi_comm.c` GPU-aware MPI path (plan §11 risk #5).** No current
   test exercises raw device pointers through `MPI_Send`/`MPI_Recv`. If a
   future workflow does, set `MPICH_GPU_SUPPORT_ENABLED=1` for that
   workflow only — see `phase5_handoff.md` "Environment" note.

4. **`MPICH_SMP_SINGLE_COPY_MODE=NONE` audit on production Frontier.** The
   dev sub-system requires this for `nccl_bcast_*_host` tests because
   XPMEM kernel module access is restricted. Production Frontier nodes
   typically have XPMEM accessible — verify on the real target before
   baking the flag into a permanent recipe.

5. **Simulation-app shakedown.** The plan's §10 noted "green light to
   start porting the simulation apps in a later phase" once the bring-up
   sequence passed. With Phase 6 complete and 16 of 18 unit tests from
   `mctest_nccl_comm` passing (including all sync, per_sync, multicomm,
   broadcast, allgather variants), the foundation is in place. The
   `vlasov/`, `gyrokinetic/`, `pkpm/` directories will need their own
   `_cu.cu` + `GKYL_HAVE_CUDA → GKYL_HAVE_GPU` sweeps following the same
   Phase 4 pattern, plus per-app gate-widening of any LU-dependent or
   nccl-dependent `.c` files.

### Files NOT to touch as follow-up to the port

- `core/zero/cusolver_ops.cu`, `core/zero/cudss_ops.cu` — out of scope
  per plan §7.
- `core/unit/ctest_cusolver.cu`, `core/unit/ctest_cudss.cu` — CUDA-only.

### Where to find things

- All shims live in `core/zero/`:
  - [`gkyl_gpu_runtime.h`](gkeyll-agent-dev/core/zero/gkyl_gpu_runtime.h) (Phase 1; CUDA Runtime ↔ HIP runtime, plus the `__CUDA_ARCH__` alias added in Phase 5)
  - [`gkyl_gpu_reduce.h`](gkeyll-agent-dev/core/zero/gkyl_gpu_reduce.h) (Phase 3; CUB ↔ hipCUB)
  - [`gkyl_gpu_blas.h`](gkeyll-agent-dev/core/zero/gkyl_gpu_blas.h) (Phase 5; cuBLAS ↔ rocBLAS+rocSOLVER)
  - **No new shim was needed for NCCL ↔ RCCL** — RCCL is symbol-compatible with NCCL out of the box, so a single conditional include in `gkyl_nccl_comm.h` was enough.
- Top-level Makefile USING_HIPCC block: [`gkeyll-agent-dev/Makefile`](gkeyll-agent-dev/Makefile)
- Sub-make: [`gkeyll-agent-dev/core/Makefile-core`](gkeyll-agent-dev/core/Makefile-core)
- Configure script: [`gkeyll-agent-dev/machines/configure.frontier-gpu.sh`](gkeyll-agent-dev/machines/configure.frontier-gpu.sh)
- Earlier handoffs: see top of this file.
- Long-form bug notes for AMD-port-relevant pitfalls live in this session's
  Claude memory dir (`~/.claude/projects/.../memory/`):
  - `array_reduce_gpu_output_pointer.md` (Phase 4)
  - `cu_dev_new_struct_inner_pointer_fixup.md` (Phase 4)
  - `cuda_arch_macro_aliasing_under_hip.md` (Phase 5)
  - `nccl_post_2_19_group_call_contract.md` (Phase 6)
- User-supplied reference: [`NCCL_MIGRATION_NOTES.md`](NCCL_MIGRATION_NOTES.md)
  (peer to this handoff in `working_dev/`).
