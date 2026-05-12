# AMD Port — Vlasov V5 Handoff

V5 (single-GPU `rt_vlasov_weibel_2x2v_p2` fiducial run) complete. The 2x2v
Weibel-instability simulation runs to its final time on AMD MI250X under
HIP/ROCm 6.2.4. Continuation of [`vlasov_v4_handoff.md`](vlasov_v4_handoff.md);
plan in [`amd_vlasov_port_plan.md`](amd_vlasov_port_plan.md).

V5 took longer than V0–V4 combined because the initial fault
(`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on the first volume kernel
launch) had a non-obvious root cause that took deep diagnostic work to
isolate. The fix turned out to be a one-file source-level refactor —
but reaching it required ruling out half a dozen plausible alternative
explanations. This handoff captures both the fix and the failed
hypotheses, since the latter inform what to look for in future
AMD-port work elsewhere in the codebase.

---

## Initial symptom

Launching `rt_vlasov_weibel_2x2v_p2.lua` faulted within the first
`gkyl_hyper_dg_advance` call:

```
Memory access fault by GPU node-4 (Agent handle: ...) on address 0x...
Reason: Unknown.
```

Behavior pattern:

- Fault was reproducible on every run, but the **fault address varied**
  (sometimes near valid device-pointer ranges, sometimes apparent host
  addresses, sometimes NULL). Address-pattern non-determinism is
  consistent with a wild dereference of garbage data.
- Fault appeared on **first kernel call** of the time-step loop. The
  `[KDBG hyper_dg STAGE0] entered kernel` printf added at the very top
  of `gkyl_hyper_dg_advance_cu_kernel` did **not** fire on weibel —
  consistent with the kernel's printf ring buffer never flushing
  because the kernel aborted before completing.
- The same printf fired for the V4 1x1v twostream sim, indicating the
  problem was specific to the 2x2v p2 path, not generic to hyper_dg.
- 2x2v **p1** worked. 1x1v p2 worked. Only 2x2v p2 faulted.

---

## Hypotheses ruled out (and why each was worth ruling out)

Each of these took at least one rebuild + rerun cycle. Documenting them
because each one tested a real, independently-plausible mechanism for
this class of fault, and the same hypotheses will arise next time a
similar fault appears on another kernel.

### 1. Was it `-ffast-math` reassociation?

Hypothesis: `-ffast-math` enables FMA reassociation on AMDGPU which
historically has had codegen issues with deeply-nested FMA trees in
`out[i] += a*b + c*d + ...` patterns.

Test: clean rebuild without `-ffast-math`. Result: **same fault, same
pattern.** Hypothesis ruled out. (Side observation: `-ffast-math` is
mildly *anti-helpful* on the larger 32²×16² grid — increases register
pressure ~7%, see "perf characterization" below — but doesn't cause the
fault.)

### 2. Was it threads-per-block / occupancy?

Hypothesis: AMDGPU register allocator might choose poorly at 256
threads/block on this kernel. Reducing threads/block could give each
thread more scratch.

Test: dropped `GKYL_DEFAULT_NUM_THREADS` from 256 to 64, clean rebuild.
Result: **same fault.** Hypothesis ruled out. Confirms scratch is
allocated per-workitem regardless of block size.

### 3. Was it an LLVM 17/18 (ROCm 6.2.4) backend bug?

Hypothesis (user-supplied): "On hipcc/clang, `-ffast-math` enables
`-menable-unsafe-fp-math`, which on AMDGPU lets the backend do
aggressive FMA contraction. The AMDGPU backend in LLVM 17/18 (ROCm 6.2)
has open bugs around this for long unrolled expressions."

Test: full module-swap to `rocm/7.2.0` + `amd/7.2.0` (LLVM 22). Required
several environment fixes:
- `module swap PrgEnv-cray PrgEnv-amd` (load is silently a no-op when
  PrgEnv-cray is already loaded; swap actually takes effect).
- `module swap amd amd/7.2.0` so the compiler module matches the rocm
  library version.
- ROCm 7.2.0's hipcc clang-22 segfaulted with a Register-Coalescer-pass
  crash on `vlasov/ker/canonical_pb/canonical_pb_boundary_surfvy_3x3v_tensor_p1.c`.
  Worked around by compiling that one file at `-O1`.
- ROCm 7.2.0's HIP headers redeclare `memcpy` as `__device__`-only, which
  shadows the host `memcpy` in `vlasov/zero/velocity_map_cu.cu`'s
  `gkyl_velocity_map_new_cu_dev` (a `__host__` constructor that does
  `memcpy(gvm->vbounds, gvm_ho->vbounds, ...)`). Fixed by adding
  `#include <cstring>` to that file. Forward-compatible — kept in the
  V5 commit.
- Cray cray-mpich's `libmpi_gtl_hsa.so` is hard-linked against ROCm
  6.2.4's `libamdhip64.so.6`. Ran with both ROCm 6.2.4 and 7.2.0 lib
  paths in LD_LIBRARY_PATH so both .so versions could coexist.

Result: **same fault on weibel 2x2v p2.** Hypothesis ruled out. The bug
persists across LLVM 17/18 → LLVM 22, so it isn't an LLVM-version-bound
codegen bug.

### 4. Was it the function-pointer dispatch path?

Hypothesis: `gkyl_hyper_dg_advance_cu_kernel` calls
`up->equation->vol_term(...)` via runtime function pointer. With
`-fgpu-rdc`, the LTO linker can't see the callee at link time and must
emit a conservative ABI bridge that may interact badly with
high-register-pressure callees on AMD.

Test: introduced a direct `extern "C" __device__` wrapper
`gkyl_diag_vol_2x2v_ser_p2_direct` defined in `dg_vlasov_cu.cu`,
forward-declared in `hyper_dg_cu.cu`, and called by name when
`ndim == 4`. Result: **fault disappears.** Sim ran to completion.

Initially this looked like the answer — fn-pointer dispatch was the
problem. But further investigation (next section) showed that direct
dispatch was sidestepping the symptom rather than addressing the cause,
and that the **same fix could be achieved by reducing the callee's
spill demand** without any dispatch-path change.

---

## Diagnosis: scratch overflow at the indirect-call site

The reproducible "fault only via fn-pointer, not via direct call"
pattern, combined with knowing the affected kernel had unusually large
stack arrays (`alpha_cdim[96]` + `alpha_vdim[96]`, 1.5 KB total), led
to the actual mechanism:

- AMDGPU kernels have a single fixed `private_segment_fixed_size` per
  workitem, computed by the compiler from its visible call graph. The
  HSA runtime allocates exactly that much scratch per workitem at
  launch.
- For a **direct call**, the LTO linker sees the callee's scratch needs
  and folds them into the caller's `private_segment_fixed_size`.
  Everything fits.
- For an **indirect call** (fn-pointer), the compiler doesn't know the
  callee, so it can only reserve scratch for the caller's own state
  plus whatever fixed save area the indirect-call ABI defines. If the
  runtime callee then needs more scratch than that fits, the spill
  writes go past the workitem's slot and into the next workitem's slot
  → memory aperture violation (with apparent-random fault address,
  since the spill-target address is data-dependent on what other
  workitems were doing).

The 2x2v p2 vol kernel is the only kernel in this codebase whose
spill demand exceeds the indirect-call ABI scratch reserve. Other
kernels with similar dispatch but smaller stack arrays (1x1v p2 vol,
2x2v p1 vol, all surf kernels for any dim) stay under the threshold —
which is why the fault was 2x2v p2 vol specific.

This also explains the per-`out[i]`-bisect behavior we observed: when
we truncated the kernel body to compute only `out[1..24]`, the
kernel's max simultaneous live-value count dropped, spill volume
dropped below the ABI reserve, and the fault stopped happening. We
hadn't found a buggy line — we'd dropped under the scratch ceiling.

---

## Fix

Refactor `vlasov/ker/vlasov/vlasov_vol_2x2v_ser_p2.c`: replace the two
sparsely-used 96-element stack arrays with named `const double` scalars
for each index actually written/read (4 indices in `alpha_cdim`, 32 in
`alpha_vdim`, 36 named scalars total).

```c
// Before (768 + 768 = 1.5 KB stack arrays, mostly zero-init):
double alpha_cdim[96] = {0.0};
double alpha_vdim[96] = {0.0};
alpha_cdim[0]  = 8.0*w0dx0;
alpha_cdim[3]  = 2.309401076758503*dv0dx0;
// ... 34 more sparse writes ...
out[1] += 0.4330127018922193*(alpha_cdim[3]*f[3]+alpha_cdim[0]*f[0]);

// After (no stack arrays, named scalars in registers):
const double ac_0  = 8.0*w0dx0;
const double ac_3  = 2.309401076758503*dv0dx0;
// ... 34 more named scalars ...
out[1] += 0.4330127018922193*(ac_3*f[3]+ac_0*f[0]);
```

Generated by a 30-line Python script
(`/tmp/refactor_vol_2x2v_p2.py`) doing two passes: replace
`alpha_*dim[N] = expr` with `const double {ac,av}_N = expr` at first-write
sites; replace remaining `alpha_*dim[N]` reads with the matching scalar
name. The kernel's polynomial structure is preserved exactly — same
floating-point operations in the same order, same final result.

### Verification of the fix mechanism

Compiler resource report (`llvm-readelf --notes` on the device ELF
extracted from `libg0vlasov.so`):

```
gkyl_hyper_dg_advance_cu_kernel:
  before refactor: VGPR=128, V_spill=(non-zero), scratch=2060
  after refactor:  VGPR=176, V_spill=0,           scratch=2052
```

Compiler chose to use *more* registers (176 vs 128 — moves from 2 wave/CU
to 1 wave/CU on gfx90a) **and zero VGPR spills** rather than spill-heavy
allocation. With no register spills, the kernel's scratch demand fits
within the indirect-call ABI reserve, and **the fault no longer occurs
even with plain fn-pointer dispatch**:

```
$ gkeyll rt_vlasov_weibel_2x2v_p2_small.lua  # via eqn->vol_term fn-ptr
Number of update calls 1441
Number of forward-Euler calls 4323
RK stage-2 failures 0
Total updates took 61.4 secs
exit=0
```

### Why this is the right fix (not the dispatch-path workaround)

The direct-dispatch wrapper (`gkyl_diag_vol_2x2v_ser_p2_direct`) was a
workaround that sidestepped the symptom by exposing the callee to LTO.
The alpha-scalar refactor addresses the root cause — the kernel no
longer demands more scratch than the indirect-call ABI provides,
regardless of how it's dispatched. No architectural change to
`gkyl_dg_eqn`'s function-pointer dispatch is needed.

This matters for upstream-ability: the alpha-scalar refactor is a
strict improvement on the CUDA path too (eliminates 1.5 KB of stack
allocation that was ~half-zeroed-and-unused). The direct-dispatch
wrapper would have been a HIP-specific workaround.

---

## Performance characterization (V5)

After the fix, `rt_vlasov_weibel_2x2v_p2` (16⁴ phase grid):

```
Wall: 1:34
Total updates: 59.6 sec  (1441 steps)
Species RHS: 41.6 sec     (vol + surf)
Field RHS:    2.3 sec     (Maxwell)
```

At the larger 32²×16² grid:

```
Wall: 6:38
Total updates: 398 sec    (2882 steps)
Species RHS: 311 sec
Field RHS:    4.5 sec
```

Compared to A100 baseline at 32²×16² (Total updates 142.6 sec): MI210
is **~2.8× slower** in the compute-bound regime. The smaller V4
twostream test was ~24× slower, which appears to be launch-overhead-bound
rather than compute-bound — confirmed by the near-linear scaling we
saw at 1024×1024 twostream (~60 sec/step on AMD vs ~3 sec total on A100
at the smaller grid).

### Diagnosed mechanism for the remaining 3× gap

Same as the V5 fault root cause, just in a non-faulting regime:
register pressure on the dispatching kernels keeps occupancy at 1
wave/CU on gfx90a (VGPR=128 for the array-form kernel, 176 after the
refactor). With low occupancy, the GPU can't hide HBM latency between
spill/reload round-trips. Reducing register pressure further (the
remaining `mom_calc`'s `momLocal[96]`, the `alpha_cdim[20]` /
`alpha_vdim[20]` arrays in the surf kernels) would close more of the
gap, but those reductions are out of scope for V5 — V5's deliverable
is "weibel runs", which it does.

### Optional auxiliary refactor

We tested adding `__restrict__` to all input pointers in
`vlasov_vol_2x2v_ser_p2` (currently only `out` has it). Result on the
already-alpha-scalar-refactored kernel: additional ~7% speedup on
weibel 32²×16² (Species RHS 305 sec → 311 sec without restrict) by
allowing the compiler to keep `f[N]` values cached in registers across
`out[]` writes. **Not committed** — restrict needs to be applied
uniformly across the whole `vlasov/ker/vlasov/` family for consistency,
which is broader than V5's scope. Logged as a follow-up.

---

## V5 changeset

```
vlasov/ker/vlasov/vlasov_vol_2x2v_ser_p2.c   alpha array → named scalars
                                              (eliminates 1.5 KB stack
                                               allocation; fixes the
                                               indirect-call scratch-
                                               overflow fault)
vlasov/zero/velocity_map_cu.cu                #include <cstring> — host
                                              memcpy needed under newer
                                              HIP headers; harmless under
                                              ROCm 6.2.4. Forward-compatible.
```

The V5 fix is a single source file (the kernel). The cstring include
is an unrelated portability fix surfaced during the ROCm 7.2.0 hypothesis
test and kept because it's a strict improvement.

---

## V6 — Next (multi-GPU fiducial)

Per the plan, V6 runs one of the two fiducials across two ranks on a
single node:

```bash
salloc -A fus183 -N 1 --gpus 2 -t 30:00
srun -n 2 -c 8 --gpus-per-task=1 \
     hip-build/gkeyll/gkeyll vlasov/luareg/rt_vlasov_weibel_2x2v_p2.lua
```

Multi-GPU exercises:
- RCCL collectives between ranks (already validated in core Phase 6).
- Cray MPICH GPU-aware MPI (`libmpi_gtl_hsa.so`) **if** the Vlasov app
  passes raw device pointers through `MPI_Send`/`MPI_Recv`. If V6
  fails with intra-node weirdness, set `MPICH_GPU_SUPPORT_ENABLED=1`
  for that specific run.
- Domain decomposition / ghost-cell exchange across ranks. The single-
  rank V4/V5 paths exercise ghost-cell exchange within a rank only.

---

## Open follow-ups (not blocking V6)

1. **Apply `__restrict__` uniformly** across `vlasov/ker/vlasov/`. Tested
   on `vlasov_vol_2x2v_ser_p2.c` for ~7% additional speedup; needs
   uniform coverage across all generated vol/surf/boundary_surf
   kernels (and likely `mom_calc` kernels too) for full benefit.
   Best done in the kernel generator (`gkyl-gen`) since these files are
   generated, not hand-written.

2. **Apply alpha-scalar refactor to other vol kernels** with similar
   sparsely-used large stack arrays. The V5 fault hit
   `vlasov_vol_2x2v_ser_p2.c` first; other large generated kernels
   (`vlasov_vol_2x3v_*_p2`, `vlasov_vol_3x3v_*_p1/p2`, `canonical_pb_vol_*`,
   tensor-basis variants) have similar shape and may be on the danger
   side of the same scratch-overflow threshold. Best done in
   `gkyl-gen`.

3. **`mom_calc_advance_cu_ker`'s `momLocal[96]`**. Same pattern as
   the alpha arrays — hard-coded for the maximum confBasis size, mostly
   unused. Profiling shows 22% of GPU time in `mom_calc` for the
   weibel sim. Refactoring it to a smaller per-(cdim, vdim, poly_order)
   sized array would reduce register pressure further.

4. **Fault-mechanism audit.** The same scratch-overflow-at-indirect-call
   pattern can hit any kernel whose spill demand exceeds the ABI
   reserve. We hit it first on 2x2v p2 vol; other large kernels in
   `vlasov/ker/`, `gyrokinetic/ker/`, `pkpm/ker/` are at risk
   under HIP. Audit suggestion: any generated kernel with stack arrays
   ≥1 KB (sparsely-used or dense) is a candidate. Forced-direct-dispatch
   is a HIP-specific workaround we want to avoid; alpha-scalar refactor
   is the upstream-able fix.

5. **Memory note added.** `frontier_hipcc_module_setup.md` was added to
   the agent memory store during V5 (login-node module commands don't
   persist across Bash sub-shells; `module load PrgEnv-amd` is silently
   a no-op when PrgEnv-cray is already loaded — must use `module swap`).

---

## Where to find things

- This handoff: `/autofs/nccsopen-svm1_home/junoravin/working_dev/vlasov_v5_handoff.md`
- Earlier vlasov handoffs: [`vlasov_v0_v1_handoff.md`](vlasov_v0_v1_handoff.md),
  [`vlasov_v2_handoff.md`](vlasov_v2_handoff.md),
  [`vlasov_v3_handoff.md`](vlasov_v3_handoff.md),
  [`vlasov_v4_handoff.md`](vlasov_v4_handoff.md)
- Vlasov port plan: [`amd_vlasov_port_plan.md`](amd_vlasov_port_plan.md)
- Core port plan and handoffs: [`amd_port_plan.md`](amd_port_plan.md)
  through [`phase6_handoff.md`](phase6_handoff.md)
- Long-form bug notes in `~/.claude/projects/.../memory/`:
  `array_reduce_gpu_output_pointer.md`, `cu_dev_new_struct_inner_pointer_fixup.md`,
  `cuda_arch_macro_aliasing_under_hip.md`, `nccl_post_2_19_group_call_contract.md`,
  `frontier_hipcc_module_setup.md`
