# AMD Port — Phase 4 Handoff

This file captures Phase 4 completion state and the Phase 5 plan, so a fresh
Claude session can pick up cleanly.

Source plan: [`amd_port_plan.md`](amd_port_plan.md). Earlier handoffs:
[`phase1_handoff.md`](phase1_handoff.md), [`phase2_handoff.md`](phase2_handoff.md),
[`phase3_handoff.md`](phase3_handoff.md). Read plan §6a (LU dependency),
§8 (unit tests), and §10 (phasing) before starting Phase 5.

---

## Environment

Unchanged from Phase 3: `PrgEnv-amd` + `rocm/6.2.4` + `craype-accel-amd-gfx90a`
+ `cray-mpich` + `cray-libsci`. Working tree
`/autofs/nccsopen-svm1_home/junoravin/working_dev/gkeyll-agent-dev/`. Account
`fus183`. Every Bash invocation prefixes the module-load + `MPICH_GPU_SUPPORT_ENABLED=1`
+ `LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH` line.

---

## Phase 4 — Complete (with documented carve-outs)

**Headline:** every `_cu.cu` file in `core/zero/` (17 of them) compiles cleanly
under hipcc and links into `libg0core.so` with `-fgpu-rdc`. The broader
non-LU unit-test surface passes on AMD.

```text
hip-build/core/libg0core.so   25.7 MB   (was 25.6 MB at end of Phase 3)
  17 *.cu.o objects in zero/  (unchanged inventory; gates flipped, not file count)
   6 *.cu.o objects in unit/  (unchanged from Phase 3)
```

### Gate test results (AMD MI250X, gfx90a, ROCm 6.2.4)

| Test | Result | Notes |
|---|---|---|
| `ctest_array_ops` | SUCCESS | Tier-1 arithmetic. |
| `ctest_dg_array_mask` | 49/49 | Required two real fixes — see §"Bugs found and fixed" below. |
| `ctest_struct_of_arrays` | SUCCESS | |
| `ctest_tensor_field_ops` | 12/12 | Required a real fix — see §"Bugs found and fixed". |
| `ctest_tensor_field` | SUCCESS | Construction + accessors only; no kernel. |
| `ctest_basis` | SUCCESS | Drives `ctest_basis_cu.cu` device code via host driver. |
| `ctest_dg_basis_ops` | SUCCESS | |
| `ctest_skin_surf_from_ghost` | SUCCESS | |
| `ctest_dg_bin_ops` | 15/24 | The 9 failing `test_*d_p*_cu` tests trace to `gkyl_nmat_cu_dev_new` / `gkyl_nmat_linsolve_lu_pa` — **Phase 5-blocked** per plan §6a. |
| `ctest_array_average` | 3/6 | The 3 failing `test_*x_gpu` tests trace to the same `gkyl_nmat_cu_dev_new` assert-stub — **Phase 5-blocked**. |

### Carve-outs documented in plan

The Phase 4 row in `amd_port_plan.md` §10 was amended to acknowledge the LU
carve-outs explicitly:

> Gate: `ctest_array_ops` passes; the §6a-tracked LU-dependent paths in
> `ctest_dg_bin_ops` and `ctest_array_average` remain failing on AMD until
> Phase 5 ports `gkyl_nmat_cu_dev_new`/`gkyl_nmat_linsolve_lu_pa` to rocBLAS
> — both call sites assert-stub through `mat.c` under HIP today.

Plan §8c was amended to note the five new smoke tests for `dg_geom`,
`dg_interpolate`, `nodal_ops`, hybrid/gkhybrid bases are deferred — the
underlying `_cu.cu` files all compile under hipcc, but authoring the
runtime smoke tests has been pushed out (likely paired with simulation-app
shakedown). See the new "Deferred to a later phase" callout in §8c.

### Bugs found and fixed (real bugs in the codebase, surfaced by AMD bring-up)

1. **`ctest_dg_array_mask` — host stack pointer fed to GPU reduce kernel.**
   Every failing GPU test had the pattern
   `double global_max; gkyl_array_reduce(&global_max, gpu_arr, GKYL_MAX);`.
   `gkyl_array_reduce`'s GPU path atomic-writes the result through the output
   pointer, which on AMD MI250X must be device-accessible (or pinned-host).
   Plain stack address faults at exactly that address (`0x7fffffff6000` in
   the failing runs). Fix: added a `reduce_max_to_host` helper at the top
   of [`core/unit/ctest_dg_array_mask.c`](gkeyll-agent-dev/core/unit/ctest_dg_array_mask.c)
   that routes the output through `gkyl_cu_malloc` + `gkyl_cu_memcpy` D2H
   when `use_gpu`. 17 call sites updated. Mirrors the pattern already in
   [`ctest_array_reduce.c`](gkeyll-agent-dev/core/unit/ctest_array_reduce.c).

2. **`ctest_tensor_field_ops` — `cu_dev_new` left a host pointer in the
   device clone.** [`core/zero/tensor_field.c:114`](gkeyll-agent-dev/core/zero/tensor_field.c#L114)
   was bulk-memcpying the host struct to device, then "fixing up" `tdata`
   with `&tfld->tdata` — but that's the host `gkyl_array*`, not the device
   one. So on-device `met->tdata->...` dereferences a host pointer and
   faults at a low/heap-like address (`0xa0f000` in the failing runs). Fix:
   change source to `&tfld->tdata->on_dev` so the device-resident `tdata`
   field holds the device gkyl_array pointer. Same pattern as
   [`dg_array_mask_cu.cu:cu_dev_new`](gkeyll-agent-dev/core/zero/dg_array_mask_cu.cu).

   Two fix patterns are valid for this class of issue and both are documented
   in the long-form memory note `cu_dev_new_struct_inner_pointer_fixup.md`:
   either an explicit per-field fix-up after the bulk memcpy, or a
   swap-restore around the bulk memcpy. The tensor_field fix uses pattern 1;
   dg_array_mask uses pattern 2.

3. **Red-herring detour worth flagging.** Initial debugging of (1) led down
   a multi-hour rabbit hole hypothesizing that HIP/clang's calling convention
   silently degrades large by-value `struct gkyl_range` kernel arguments to
   host-pointer-byref above some size threshold. That hypothesis is *probably
   wrong* — embed-by-value mods to the dg_array_mask struct didn't change
   the symptom, and the actual fault was always upstream in
   `gkyl_array_reduce`. The embed-by-value churn was reverted; only the
   `GKYL_HAVE_CUDA → GKYL_HAVE_GPU` sweep remains. **If a future similar
   fault appears, audit the test for host-pointer-to-GPU-kernel patterns
   *first*, before assuming a calling-convention issue.**

### Files modified in Phase 4

#### Host gate sweep (`GKYL_HAVE_CUDA` → `GKYL_HAVE_GPU`)

Per Phase 3 handoff work item #2. All 12 unit-test drivers and a wider set
of `core/zero/` files were swept so the GPU branches actually compile and
dispatch under HIP rather than falling through to assert-stubs.

`core/unit/`: `ctest_array_average.c`, `ctest_array_ops.c`, `ctest_basis.c`,
`ctest_dg_array_mask.c`, `ctest_dg_basis_ops.c`, `ctest_dg_bin_ops.c`,
`ctest_range.c`, `ctest_rect_grid.c`, `ctest_skin_surf_from_ghost.c`,
`ctest_struct_of_arrays.c`, `ctest_tensor_field.c`, `ctest_tensor_field_ops.c`.

`core/zero/`: `array_average.c`, `array_ops.c`, `cart_modal_gkhybrid.c`,
`cart_modal_hybrid.c`, `cart_modal_serendip.c`, `cart_modal_tensor.c`,
`dg_array_mask.c`, `dg_basis_ops.c`, `dg_bin_ops.c`, `dg_geom.c`,
`dg_interpolate.c`, `eval_on_nodes.c`, `nodal_ops.c`,
`skin_surf_from_ghost.c`, `tensor_field.c`, `tensor_field_ops.c`,
plus the priv headers `gkyl_array_average_priv.h`,
`gkyl_dg_array_mask_priv.h`, `gkyl_dg_basis_ops_priv.h`,
`gkyl_dg_interpolate_priv.h`, `gkyl_skin_surf_from_ghost_priv.h`.

#### Stack-frame fix in `core/unit/ctest_basis_cu.cu`

`ker_dev_cu_ser_2d` used `double z[128], b[128]` and `double nodes[128*128]`.
The `[128*128]` allocation alone is 128 KB, putting the kernel's per-function
stack at 132 KB — exceeds AMDGPU's 131 056-byte hard limit. Resized to
the values actually asserted (`z[GKYL_MAX_DIM], b[32], nodes[64]`). Now
links cleanly under hipcc and produces correct results under nvcc as well
(NVIDIA's default ~1 KB per-thread reservation no longer spills to local
memory). The Phase 3 handoff flagged this as the first Phase 4 deliverable.

#### Real bugfixes (see §"Bugs found and fixed" above)

`core/unit/ctest_dg_array_mask.c` (added `reduce_max_to_host` helper, 17
call sites updated). `core/zero/tensor_field.c` (`cu_dev_new` inner-pointer
fix-up).

### How to rebuild + run from scratch

```bash
cd /autofs/nccsopen-svm1_home/junoravin/working_dev/gkeyll-agent-dev
module load PrgEnv-amd rocm/6.2.4 craype-accel-amd-gfx90a cray-mpich cray-libsci
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

rm -rf hip-build
make core -j 8
make core-unit -j 8

# Phase 4 gate tests (compute node, 5-min reservation each):
for t in ctest_array_ops ctest_dg_array_mask ctest_struct_of_arrays \
         ctest_tensor_field_ops ctest_tensor_field ctest_basis \
         ctest_dg_basis_ops ctest_skin_surf_from_ghost; do
  srun -A fus183 -p batch -N 1 -n 1 -c 8 --gpus-per-task=1 -t 5:00 \
      hip-build/core/unit/$t
done

# Expected partials (Phase 5-blocked LU dependency):
#   ctest_dg_bin_ops      — 15/24, 9 *_cu tests SIGABRT in gkyl_nmat_cu_dev_new
#   ctest_array_average   — 3/6,   3 _gpu tests SIGABRT in gkyl_nmat_cu_dev_new
```

---

## Phase 5 — Next

**Goal:** port `gkyl_nmat_cu_dev_new`, `cu_nmat_linsolve_lu`, and
`gkyl_nmat_linsolve_lu_pa` from cuBLAS batched LU to rocBLAS batched LU.
Per plan §6a, the public entry point is called from
[`dg_bin_ops_cu.cu:355`](gkeyll-agent-dev/core/zero/dg_bin_ops_cu.cu#L355) and
[`dg_bin_ops_cu.cu:446`](gkeyll-agent-dev/core/zero/dg_bin_ops_cu.cu#L446)
(the `gkyl_dg_div_op_cu` / `gkyl_dg_div_op_range_cu` paths). Both
`ctest_dg_bin_ops` and `ctest_array_average` will go green once these land.

**Gate (per plan §10):** `ctest_mat` passes on AMD, and the previously
Phase-5-blocked tests (`ctest_dg_bin_ops` 9-of-24, `ctest_array_average`
3-of-6) flip to green.

### Work items for Phase 5

1. **Create the BLAS shim.** Mirror the `gkyl_gpu_runtime.h` (Phase 1) and
   `gkyl_gpu_reduce.h` (Phase 3) patterns: a single header
   `core/zero/gkyl_gpu_blas.h` that resolves to `<cublas_v2.h>` /
   `<cusolverDn.h>` on CUDA and `<rocblas/rocblas.h>` (and possibly
   `<rocsolver/rocsolver.h>`) on HIP. Wrap the HIP arm in `extern "C++"`
   following the same defense as the runtime / reduce shims.

2. **Audit `core/zero/mat.c` and `core/zero/gkyl_mat_priv.h`** for every
   `cublas*` / `cusolver*` symbol. The identified hot path is the batched
   LU pair — verify whether single-matrix LU, dot/axpy/gemm, or any
   cuSOLVER wrappers are also touched. Keep the existing `mat.c` cuBLAS
   path intact under `USING_NVCC`; add a `USING_HIPCC` arm that calls the
   rocBLAS equivalents.

3. **Flip the `mat.c` gate** from `#ifdef GKYL_HAVE_CUDA` to
   `#ifdef GKYL_HAVE_GPU` once the rocBLAS arm is in place. Same pattern
   the Phase 3 handoff documented for the reduction host files.

4. **Watch for rocBLAS handle lifetime quirks.** rocBLAS handle creation
   is asynchronous and the first call has measurable startup cost; the
   existing CUDA path may not allocate a long-lived handle. If `mat.c`
   creates a handle per call, that's a perf regression on AMD — consider
   the static-handle pattern cholla uses.

5. **Re-run the Phase 4 gate tests** to confirm `ctest_dg_bin_ops` and
   `ctest_array_average` flip from partial to full pass:
   ```bash
   srun -A fus183 -p batch -N 1 -n 1 -c 8 --gpus-per-task=1 -t 5:00 \
       hip-build/core/unit/ctest_mat
   srun -A fus183 -p batch -N 1 -n 1 -c 8 --gpus-per-task=1 -t 5:00 \
       hip-build/core/unit/ctest_dg_bin_ops
   srun -A fus183 -p batch -N 1 -n 1 -c 8 --gpus-per-task=1 -t 5:00 \
       hip-build/core/unit/ctest_array_average
   ```

### Deferred to later phases (track separately)

- **§8c smoke tests for `dg_geom`, `dg_interpolate`, `nodal_ops`,
  hybrid/gkhybrid bases.** Pushed out of Phase 4 (see plan §8c note).
  Likely added during simulation-app shakedown. Compile-side is already
  green — `hip-build/core/zero/{cart_modal_hybrid,cart_modal_gkhybrid,
  dg_geom,dg_interpolate,nodal_ops}_cu.cu.o` all build.
- **Phase 6 — RCCL via macro switch.** Single-node `mctest_nccl_comm`
  pass.

### Files NOT to touch in Phase 5

- `core/zero/cusolver_ops.cu`, `core/zero/cudss_ops.cu` — out of scope
  per plan §7 (linear-solver porting deliberately not on AMD).
- `core/unit/ctest_cusolver.cu`, `core/unit/ctest_cudss.cu` — CUDA-only,
  excluded via `HIP_DEFER_UNIT` in `core/Makefile-core`.
- `core/zero/nccl_comm.c`, `core/zero/multib_comm_conn_nccl.c` — Phase 6.

### Where to find things

- Phase 1 runtime shim: [`gkeyll-agent-dev/core/zero/gkyl_gpu_runtime.h`](gkeyll-agent-dev/core/zero/gkyl_gpu_runtime.h)
- Phase 3 reduction shim: [`gkeyll-agent-dev/core/zero/gkyl_gpu_reduce.h`](gkeyll-agent-dev/core/zero/gkyl_gpu_reduce.h)
- Top-level Makefile USING_HIPCC block: [`gkeyll-agent-dev/Makefile`](gkeyll-agent-dev/Makefile)
- Sub-make: [`gkeyll-agent-dev/core/Makefile-core`](gkeyll-agent-dev/core/Makefile-core)
- Earlier handoffs: [`phase1_handoff.md`](phase1_handoff.md),
  [`phase2_handoff.md`](phase2_handoff.md), [`phase3_handoff.md`](phase3_handoff.md)
- Long-form bug notes for AMD-port-relevant pitfalls live in this session's
  Claude memory dir (`~/.claude/projects/.../memory/`) — two notes saved
  in Phase 4: `array_reduce_gpu_output_pointer.md` (host stack as GPU
  reduce output) and `cu_dev_new_struct_inner_pointer_fixup.md` (host
  pointer left in device clone).

### Deferred from Phase 4 (track separately)

- Five §8c smoke tests (see "Deferred to later phases" above).
- Audit of remaining `GKYL_CU_D static ... choose_kernel` host-only
  dispatcher patterns — Phase 4 surface (the priv headers swept above)
  surfaced none. Same per-phase grep discipline applies to Phase 5's
  `mat.c` work.
