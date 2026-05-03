# AMD Port — Phase 3 Handoff

This file captures Phase 3 completion state and the Phase 4 plan, so a fresh
Claude session can pick up cleanly.

Source plan: [`amd_port_plan.md`](amd_port_plan.md). Earlier handoffs:
[`phase1_handoff.md`](phase1_handoff.md), [`phase2_handoff.md`](phase2_handoff.md).
Read plan §8 (unit tests, including §8b bring-up sequence and §8c coverage
gap) and §10 (phasing) before starting Phase 4.

---

## Environment

- Host: `login1` of the Frontier dev sub-system. Account `fus183`.
- Working dir: `/autofs/nccsopen-svm1_home/junoravin/working_dev/`. Source tree
  under `gkeyll-agent-dev/`. Reference cholla port under `cholla/`.
- Toolchain: `PrgEnv-amd` + `rocm/6.2.4` + `craype-accel-amd-gfx90a` +
  `cray-mpich` + `cray-libsci` (libsci_amd). hipCUB ships at
  `/opt/rocm-6.2.4/include/hipcub/hipcub.hpp`.
- Bash tool calls do **not** persist module loads. Every build invocation
  must prefix `module load PrgEnv-amd rocm/6.2.4 craype-accel-amd-gfx90a
  cray-mpich cray-libsci` (and export `MPICH_GPU_SUPPORT_ENABLED=1` plus
  `LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH`) inline.

---

## Phase 3 — Complete

**Gates met:**

| Test | Result | Notes |
|---|---|---|
| `ctest_array_reduce` | 10/10 | Includes `cu_array_reduce_max`, `cu_array_reduce_max_big`, `cu_array_reduce_range_{1d,2d}_max`, plus two `cu_array_reduce_range_max_timer` runs at 32×32×40×40 and 32×32×32×32. |
| `ctest_array_dg_reduce` | 4/4 | Host + GPU DG-component reductions. |
| `ctest_array_integrate` | 8/8 | 1x/2x integrate + gradsq, host + GPU. |

These are the warp-vs-wave64 canaries (plan §5d/§5g). All green proves
hipCUB BlockReduce produces correct results on gfx90a's wave64 hardware
with `BLOCKSIZE = GKYL_DEFAULT_NUM_THREADS = 256` (4 waves/block, vs.
NVIDIA's 8 warps/block).

```text
hip-build/core/libg0core.so  25.6 MB  (was 25 MB at end of Phase 2)
  17 *.cu.o objects in zero/  (was 14 — added array_reduce_cu, array_dg_reduce_cu, array_integrate_cu)
   6 *.cu.o objects in unit/  (unchanged from Phase 2)
```

### Files created

- `core/zero/gkyl_gpu_reduce.h` — hipCUB / CUB translation shim per plan §5c.
  Single include point that resolves to `<cub/cub.cuh>` on CUDA and
  `<hipcub/hipcub.hpp>` on HIP, with `namespace cub = hipcub;` alias on the
  HIP side so `cub::BlockReduce<...>`, `cub::Max()`, `cub::Min()`, `cub::Sum()`
  template instantiations resolve unchanged. The HIP arm is wrapped in
  `extern "C++"` so it parses cleanly even when reached transitively from
  inside an `extern "C" {}` block (the same defense as in
  `gkyl_gpu_runtime.h`). HIP arm errors at compile time if it is ever
  reached from a non-C++ TU — this shim is for `.cu` files only.

### Files modified

- `core/zero/array_reduce_cu.cu`,
  `core/zero/array_dg_reduce_cu.cu`,
  `core/zero/array_integrate_cu.cu` —
  changed `#include <cub/cub.cuh>` → `#include <gkyl_gpu_reduce.h>`.
  Bodies untouched: `cub::BlockReduce<double, BLOCKSIZE>`,
  `cub::Max()`, `cub::Min()`, `cub::Sum()`,
  `__shared__ typename BlockReduceT::TempStorage temp`,
  the hand-rolled `atomicMax_double` / `atomicMin_double` block — all
  resolve via the namespace alias.
- `core/zero/array_reduce.c`,
  `core/zero/array_dg_reduce.c`,
  `core/zero/array_integrate.c` —
  swept `GKYL_HAVE_CUDA` → `GKYL_HAVE_GPU`. These were intentionally
  left on `GKYL_HAVE_CUDA` at the end of Phase 2 so their assert-stubs
  satisfied link references; with the `_cu.cu` impls now compiling under
  HIP, the gates flip and the host code dispatches to the real
  device-side functions.
- `core/zero/gkyl_array_reduce_priv.h`,
  `core/zero/gkyl_array_dg_reduce_priv.h` —
  same sweep so the `_cu` function declarations are visible under HIP.
  (`gkyl_array_integrate_priv.h` had no `GKYL_HAVE_CUDA` gates — only
  `GKYL_CU_D` lookup tables, which are correct as-is.)
- `core/unit/ctest_array_reduce.c`,
  `core/unit/ctest_array_dg_reduce.c`,
  `core/unit/ctest_array_integrate.c` — same sweep so the GPU tests run
  under HIP. (Without this, the binary would build and silently no-op
  the GPU portion, mirroring the Phase 2 ctest_alloc.c / ctest_array.c
  finding.)
- `core/Makefile-core` — removed the `HIP_DEFER_REDUCE` variable; the
  three reduction `_cu.cu` files are now picked up by the regular
  `find $(SRC_DIRS) -name *.cu` shell. The `HIP_DEFER_UNIT` filter
  remains (still defers `ctest_cusolver.cu`, `ctest_cudss.cu`,
  `ctest_basis_cu.cu`).

### Plan §5f verified — `CUDART_VERSION` is undefined under hipcc

```bash
hipcc -E -dM -x hip - </dev/null | grep -i CUDART
# (no output)
```

This means the `#if CUDART_VERSION > 12090` blocks in the reduction
files (which use the CUDA-12.9+-specific `::cuda::maximum` /
`::cuda::minimum` / `::cuda::std::plus` functors that hipCUB does NOT
provide) automatically fall to the `cub::Max()` / `cub::Min()` /
`cub::Sum()` branch — which hipCUB DOES support via the namespace alias.
No defensive `!defined(GKYL_HAVE_HIP)` extension is required on
Frontier with ROCm 6.2.4.

If a future toolchain refresh leaks `CUDART_VERSION` (e.g., a hipcc
build that bundles a CUDA SDK on the include path), harden the guards
to `#if defined(CUDART_VERSION) && CUDART_VERSION > 12090 && !defined(GKYL_HAVE_HIP)`.

### API design audit (Phase 2 finding extended)

Greppped the reduction priv headers for the host-only-dispatcher
pattern (`static GKYL_CU_D ... choose_kernel ... if (use_gpu) ... cu(...)`)
that broke `gkyl_skin_surf_from_ghost_priv.h` in Phase 2. **None
found.** The `GKYL_CU_D` annotations in
`gkyl_array_integrate_priv.h` are on `static const` lookup tables
(`gkyl_array_integrate_none_ker_list_ser`, `..._abs_ker_list_ser`,
`..._sq_ker_list_ser`, `..._sq_weighted_ker_list_*`,
`..._gradsq_ker_list`, `..._gradperpsq_ker_list` etc.) — these are
arrays of kernel function pointers indexed by basis dim and poly
order, consumed directly by device kernels. Correct use of the
annotation; no changes needed.

The grep-on-each-phase discipline is paying off — confirms the Phase 2
finding is contained to dispatcher functions, not present in lookup
tables.

### How to rebuild from scratch

```bash
cd /autofs/nccsopen-svm1_home/junoravin/working_dev/gkeyll-agent-dev
module load PrgEnv-amd rocm/6.2.4 craype-accel-amd-gfx90a cray-mpich cray-libsci
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

rm -rf hip-build
make core -j 8
make core-unit -j 8

# Phase 3 gate tests (compute node, 5-min reservation is plenty):
for t in ctest_array_reduce ctest_array_dg_reduce ctest_array_integrate; do
  srun -A fus183 -p batch -N 1 -n 1 -c 8 --gpus-per-task=1 -t 5:00 \
      hip-build/core/unit/$t
done
```

---

## Phase 4 — Next

**Goal:** every remaining `_cu.cu` file compiles and the broader unit-test
set passes. Close the §8c coverage gap by adding minimal smoke tests for
`dg_geom`, `dg_interpolate`, `nodal_ops`, and extending `ctest_basis_cu`
to cover hybrid/gkhybrid bases.

**Gate (per plan §10):** `ctest_array_ops`, `ctest_dg_bin_ops`, plus the
five new tests pass. Implicitly the `ctest_array_average`,
`ctest_dg_array_mask`, `ctest_struct_of_arrays`, `ctest_tensor_field_ops`
tests should also pass — they exercise `_cu.cu` files that already
compile in Phases 2/3 and the gates have been swept.

### Work items (per plan §8b/§8c/§10)

1. **Resolve the `ctest_basis_cu.cu` stack-frame issue.**
   `ker_dev_cu_ser_2d` exceeds AMDGPU's 131 056-byte per-function stack
   limit by ~4 KB (135 184 reported). Read the kernel and pick the
   right fix:
   - **Spill to LDS** (`__shared__`) if the buffer is per-block, not
     per-thread.
   - **Hoist to `__constant__` memory** if it's a static lookup table.
   - **Split** the kernel into multiple smaller kernels indexed by
     dim × poly_order so each instance fits.
   The kernel is in `core/unit/ctest_basis_cu.cu` (test scaffolding,
   not a production kernel — this is the easiest place to apply the
   fix). After the fix, drop `unit/ctest_basis_cu.cu` from
   `HIP_DEFER_UNIT` in `core/Makefile-core`.

2. **Run the existing Phase 2/3 unit-test gates that haven't been
   exercised yet.** Most of these binaries already build under HIP
   (because the `_cu.cu` files compile and link), but the `.c` test
   drivers may still have `GKYL_HAVE_CUDA` gates that no-op the GPU
   tests. Sweep these to `GKYL_HAVE_GPU`:
   - `core/unit/ctest_array_average.c`
   - `core/unit/ctest_array_ops.c`
   - `core/unit/ctest_basis.c`
   - `core/unit/ctest_dg_array_mask.c`
   - `core/unit/ctest_dg_basis_ops.c`
   - `core/unit/ctest_dg_bin_ops.c`
   - `core/unit/ctest_range.c`
   - `core/unit/ctest_rect_grid.c`
   - `core/unit/ctest_skin_surf_from_ghost.c`
   - `core/unit/ctest_struct_of_arrays.c`
   - `core/unit/ctest_tensor_field.c`
   - `core/unit/ctest_tensor_field_ops.c`
   (`ctest_linsolvers.c` and `ctest_mat.c` should NOT be flipped — those
   are tied to `mat.c` which is still on `GKYL_HAVE_CUDA` until Phase 5.)

3. **Run the bring-up sequence** per plan §8b (skipping the Phase 5
   `ctest_mat` and Phase 6 `mctest_nccl_comm`):
   ```bash
   for t in ctest_array_ops ctest_dg_bin_ops ctest_array_average \
            ctest_dg_array_mask ctest_struct_of_arrays \
            ctest_tensor_field_ops ctest_basis ctest_basis_cu; do
     srun -A fus183 -p batch -N 1 -n 1 -c 8 --gpus-per-task=1 -t 5:00 \
         hip-build/core/unit/$t
   done
   ```

4. **Add the five §8c smoke tests:**
   - Extend [`core/unit/ctest_basis_cu.cu`](gkeyll-agent-dev/core/unit/ctest_basis_cu.cu)
     to cover `cart_modal_hybrid_cu.cu` and `cart_modal_gkhybrid_cu.cu`
     bases (constructor + one device-side `eval` round-trip).
   - New `core/unit/ctest_dg_geom_cu.cu`: `gkyl_dg_geom_new_from_host(..., true)`,
     assert struct fields populated and one geom value matches host.
   - New `core/unit/ctest_dg_interpolate_cu.cu`: construct interpolator
     with `use_gpu=true`, advance one input array, compare against host
     on a small grid.
   - New `core/unit/ctest_nodal_ops_cu.cu`: round-trip nodal→modal→nodal
     on GPU, assert max-norm diff under tolerance.
   Each test ≈ 30-50 lines: constructor invocation, one kernel call,
   one sanity assertion. Pattern follows existing `ctest_*_cu.cu` files
   under `core/unit/`.

5. **Audit before each gate**: 60-second grep for the host-only
   dispatcher pattern in priv headers being newly exercised by Phase 4
   tests:
   ```bash
   grep -B2 'static.*GKYL_CU_D\|GKYL_CU_D$' \
       core/zero/gkyl_dg_bin_ops_priv.h \
       core/zero/gkyl_dg_basis_ops_priv.h \
       core/zero/gkyl_dg_geom_priv.h \
       core/zero/gkyl_dg_interpolate_priv.h \
       core/zero/gkyl_nodal_ops_priv.h \
       core/zero/gkyl_tensor_field_ops_priv.h \
       core/zero/gkyl_array_average_priv.h \
       core/zero/gkyl_dg_array_mask_priv.h
   ```
   For each `static GKYL_CU_D ...` hit that calls a `_cu(...)` function
   inside a `if (use_gpu)` block: drop the `GKYL_CU_D` annotation if the
   function is host-only-called (grep call sites in the corresponding
   `.c` file). For lookup tables consumed by device code, leave alone.

### Likely fall-out to handle in Phase 4

- **More stack-frame errors**: `ker_dev_cu_ser_2d` is the first one we
  hit, but `cart_modal_hybrid_cu.cu` and `cart_modal_gkhybrid_cu.cu`
  carry similar device-side lookup machinery. If new stack-frame
  errors surface during the Phase 4 tests, the same three-option fix
  (LDS / `__constant__` / kernel split) applies. Always check the
  largest-poly-order × largest-dim instantiation first — those produce
  the biggest local arrays.

- **Host-only dispatcher patterns** (Phase 2 finding): see step 5 above.
  Most likely candidates by file naming convention are
  `*_choose_kernel`, `*_set_kernel`, `*_pick_kernel` static helpers in
  priv headers. None surfaced in Phase 3 reduction headers; we don't
  know yet how many are in the broader Phase 4 surface area. Plan for
  a few but not many.

- **Stricter overload resolution surfacing in `dg_bin_ops`**: this file
  contains `gkyl_dg_div_op_cu` / `gkyl_dg_div_op_range_cu` which call
  `gkyl_nmat_linsolve_lu_pa` (defined in `mat.c`, still on
  `GKYL_HAVE_CUDA` until Phase 5). The link should resolve to the
  assert-stub from `mat.c`'s `#else` branch. The link will succeed but
  the test will fail at runtime if exercised — `ctest_dg_bin_ops` may
  partially pass if it only exercises the non-LU paths. Worth running
  to find out, then deferring whatever specifically depends on the LU.

- **`-fgpu-rdc` link-step memory pressure**: more `.cu.o` objects mean
  more device-link work. cholla hits memory limits around 100+ device
  TUs on Frontier; we're nowhere near that yet (17 + 6 = 23 device
  TUs as of end of Phase 3). Not a near-term concern but worth
  watching as Phase 4 adds the five smoke tests.

### Files NOT to touch in Phase 4

- `core/zero/mat.c`, `core/zero/gkyl_mat_priv.h` — Phase 5 (cuBLAS).
- `core/zero/nccl_comm.c`, `core/zero/multib_comm_conn_nccl.c` — Phase 6.
- `core/zero/cusolver_ops.cu`, `core/zero/cudss_ops.cu` — out of scope (plan §7).
- `core/unit/ctest_cusolver.cu`, `core/unit/ctest_cudss.cu` — CUDA-only.
- `core/unit/ctest_linsolvers.c`, `core/unit/ctest_mat.c` — Phase 5.

### Where to find things

- Phase 3 reduction shim: [`gkeyll-agent-dev/core/zero/gkyl_gpu_reduce.h`](gkeyll-agent-dev/core/zero/gkyl_gpu_reduce.h)
- Phase 2 runtime shim: [`gkeyll-agent-dev/core/zero/gkyl_gpu_runtime.h`](gkeyll-agent-dev/core/zero/gkyl_gpu_runtime.h)
- Top-level Makefile USING_HIPCC block: [`gkeyll-agent-dev/Makefile`](gkeyll-agent-dev/Makefile) (lines ~74-131)
- Sub-make: [`gkeyll-agent-dev/core/Makefile-core`](gkeyll-agent-dev/core/Makefile-core)
- Phase 1 handoff: [`phase1_handoff.md`](phase1_handoff.md)
- Phase 2 handoff: [`phase2_handoff.md`](phase2_handoff.md)

### Deferred from Phase 3 (track separately)

- `core/unit/ctest_basis_cu.cu` — still excluded via `HIP_DEFER_UNIT`
  in `core/Makefile-core` because of the 135 184-byte stack frame in
  `ker_dev_cu_ser_2d`. First Phase 4 deliverable.
- Audit of remaining `GKYL_CU_D static ... choose_kernel` dispatchers
  — see Phase 4 work item #5. Per-phase grep, not a one-shot sweep.
