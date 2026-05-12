# AMD Port — Vlasov V2 Handoff

V2 (Vlasov library build) complete. `libg0vlasov.so` builds cleanly under
HIP (~284 MB) and links against `librccl.so.1`, `libamdhip64.so.6`,
`librocblas.so.4`, `librocsolver.so.0`. All 45 `vlasov/zero/_cu.cu` files
compiled successfully via the core shims. Continuation of
[`vlasov_v0_v1_handoff.md`](vlasov_v0_v1_handoff.md); plan in
[`amd_vlasov_port_plan.md`](amd_vlasov_port_plan.md).

This handoff focuses on the **two unplanned developments** that surfaced
during V2, both manifestations of the same underlying root cause:
**clang/hipcc enforces host/device call-resolution rules more strictly
than nvcc**. Each existing CUDA path that worked under nvcc but failed
under hipcc represents a real source-level bug that nvcc happened to
tolerate; AMD is unmasking these as we add coverage.

---

## Planned V2 work (recap)

Six Makefile-vlasov edits mirroring the moments/Makefile-moments
conversion from V1 (SRCS .cu find, UNIT_CU_SRCS, EXT_INCS/EXEC_LIB_DIRS
with RCCL paths, `.cu.o` build rule using `$(GPUCXX) ... $(GPU_FLAGS)`,
kernel-override gate widened to NVCC ∨ HIPCC, lib link `else ifdef
USING_HIPCC` arm). Source-level `GKYL_HAVE_CUDA → GKYL_HAVE_GPU` sweep
across 91 files in `vlasov/zero/`, `vlasov/apps/`, `vlasov/unit/`,
`vlasov/creg/`. One file (`vlasov_lte_proj_on_basis_cu.cu`) had unused
direct cuBLAS / cuda_runtime includes — replaced with the core shims.

These were all expected from the plan. The two below were not.

---

## Unplanned development #1: 30 canonical_pb kernel files missing `GKYL_CU_DH` on definitions

### Symptom

When the Makefile-vlasov kernel-override block compiles
`vlasov/ker/canonical_pb/*.c` with `hipcc -x hip`, hipcc errored on each
of 30 files with:

```
error: __host__ function 'canonical_pb_vol_1x1v_ser_p1' cannot overload
       __host__ __device__ function 'canonical_pb_vol_1x1v_ser_p1'
```

(One error per affected function; ~30 distinct functions across the 30
files.)

### Root cause

The header `gkyl_canonical_pb_kernels.h` declared each function with
`GKYL_CU_DH`:

```c
GKYL_CU_DH double canonical_pb_vol_1x1v_ser_p1(const double *w, ...);
```

But the corresponding `.c` definition omitted the annotation:

```c
double canonical_pb_vol_1x1v_ser_p1(const double *w, ...) { ... }
```

Under hipcc with `-x hip`, `GKYL_CU_DH` expands to `__host__ __device__`.
clang/hipcc treats the two declarations as **distinct overloads with the
same C name and parameter list**, which is ambiguous and rejected. nvcc
historically merged the two into a single `__host__ __device__` definition
silently (or applied a more permissive overload resolution that didn't
flag the mismatch).

The audit before V2 had flagged exactly this — `canonical_pb 325/355`
files have `GKYL_CU_DH`. The 30 missing files were the bug. They lurked
because nvcc tolerated the mismatch and no NVIDIA workflow surfaced the
diagnostic.

### Affected files

All in `vlasov/ker/canonical_pb/`:

- 6 `canonical_pb_fluid_*` kernels (Hasegawa-Mima / Hasegawa-Wakatani
  source, subtract_zonal × 2x_ser_p1, p2).
- 24 `canonical_pb_vol_*` kernels (1D/2D/3D phase-space volume kernels in
  serendipity and tensor bases × p1/p2).

These power the canonical Poisson-bracket fluid + Vlasov code paths.
Hasegawa-Mima/Wakatani aren't on the Vlasov–Maxwell fiducial path, but
the canonical_pb_vol_* kernels are reachable from any sim that uses the
canonical-PB Vlasov path, and the build can't proceed without all of
them compiling because the kernel-override block globs the entire
directory.

### Fix

`sed`-prepended `GKYL_CU_DH` to the function definitions:

```bash
sed -i -E 's/^(double|void|int|long) ([a-zA-Z_][a-zA-Z0-9_]*\()/GKYL_CU_DH \1 \2/'
```

The fix is symmetric — header and definition now both carry the
annotation, eliminating the mismatch. CUDA builds are unaffected because
the resolved entity is identical (`__host__ __device__`) in both cases.

### Pre-existing precedent

The V1 handoff already documented an identical pattern in
`moments/ker/fem_poisson/fem_poisson_bias_plane_lhs.c`. There we
sidestepped by keeping that kernel directory's compilation NVCC-only
(since fem_poisson is fully CUDA-only on AMD anyway). For
canonical_pb we fix at the source level because the kernels ARE on the
Vlasov-Maxwell device-bitcode path.

---

## Unplanned development #2: `GKYL_CU_D` on a host-dispatcher

### Symptom

Compiling `vlasov/zero/positivity_shift_vlasov_cu.cu` under hipcc:

```
zero/gkyl_positivity_shift_vlasov_priv.h:118:5:
  error: no matching function for call to 'pos_shift_vlasov_choose_shift_kernel_cu'
```

### Root cause

The static dispatcher function in `gkyl_positivity_shift_vlasov_priv.h`
was annotated `GKYL_CU_D` (= `__device__` only):

```c
GKYL_CU_D
static void pos_shift_vlasov_choose_shift_kernel(
    struct gkyl_positivity_shift_vlasov_kernels *kernels,
    struct gkyl_basis cbasis, struct gkyl_basis pbasis,
    enum gkyl_positivity_shift_type stype,
    bool use_gpu)
{
#ifdef GKYL_HAVE_GPU
  if (use_gpu) {
    pos_shift_vlasov_choose_shift_kernel_cu(kernels, cbasis, pbasis, stype);
    return;
  }
#endif
  // ... host fallback ...
}
```

The body calls `pos_shift_vlasov_choose_shift_kernel_cu` — a host-only
launcher (declared plain `void`, internally does `<<<>>>` kernel launches
which are inherently host-only).

Under `__device__` annotation, hipcc's device pass tries to compile the
function as a device-callable entity. Inside a `__device__` function,
calls must resolve to `__device__` (or `__host__ __device__`) callees.
The `_cu` launcher is plain `__host__`, so the device pass can't find a
matching overload → "no matching function".

This is the same Phase 2 finding from the core port (handoff §5):
`static GKYL_CU_D ... choose_kernel` patterns silently break under HIP.
That sweep was per-phase; this one slipped into the Vlasov surface area.

### Why nvcc was permissive

nvcc traditionally allowed device functions to call host functions (with
varying levels of warnings depending on the version) and tended to
dead-code-eliminate aggressively when the call was gated behind a
runtime flag (`if (use_gpu)` looks like a runtime decision, but at the
device-compile-time parse stage, nvcc would let the call body parse
without forcing resolution if optimization could prove it unreachable).
Hipcc/clang is stricter: every call site in a `__device__`-annotated
function body must resolve to a `__device__`-callable target at compile
time, regardless of dead-code reachability.

### Fix

Per your guidance: change `GKYL_CU_D` to `GKYL_CU_DH` (= `__host__
__device__`) so the dispatcher is callable from both host and device
contexts. The host instantiation resolves the `_cu` call cleanly (since
the launcher is `__host__`-callable). The device instantiation is
unreachable at runtime (no device-side kernel calls this dispatcher
directly — it's exclusively used during host-side construction of the
positivity-shift updater object).

clang/hipcc allows `__host__ __device__` callers to call `__host__`-only
callees with diagnostics deferred — the function compiles in both
passes; the `if (use_gpu)` body is permitted to reference a host-only
symbol because the host instantiation is the one that's actually called.

This is the standard Gkeyll pattern for "use_gpu=true dispatcher inside
a priv header" — `GKYL_CU_DH` is the right annotation for these helpers.
Bare `GKYL_CU_D` is wrong because:
1. The function isn't actually device-callable in any current call path.
2. The `_cu` launcher it calls is fundamentally host-only.

The single occurrence in vlasov was caught here. A grep across the rest
of the tree (run as part of this fix) found no other `GKYL_CU_D static
... choose_kernel`-style dispatchers in vlasov, moments, or core priv
headers — so this is a one-shot fix, not a sweep.

---

## The pattern: AMD is unmasking real bugs nvcc tolerated

Both V2 unplanned developments fit the same shape:

- An existing CUDA path used a slightly-wrong annotation.
- nvcc accepted it (silently merged duplicate annotations, or permitted
  device-to-host calls with a warning, or dead-code-eliminated
  unreachable resolutions).
- hipcc rejects it at compile time with a hard error.

This isn't unique to V2. The core port's Phase 2 hit the same class of
finding (`gkyl_skin_surf_from_ghost_priv.h`'s host dispatcher), Phase 4
hit it inside the kernel files for `dg_array_mask` (host stack pointer
to GPU-side reduce), Phase 5 hit it via the `__CUDA_ARCH__` aliasing
issue (~99 sites silently degrading from atomic to plain `+=`), and
Phase 6 surfaced the post-2.18 NCCL group-call contract (which was
"accidentally working" via 2.18-era implicit serialization).

### Pragmatic implications for the rest of the port

1. **Each new layer surface on AMD reveals more pre-existing bugs.**
   V3 (Vlasov apps) and the future simulation-app shakedown will likely
   surface a few more annotation mismatches and wrong-gate dispatchers.
   Budget for ~1-2 small bugs per ~50-100 kernel files added to the
   compile surface, based on the V1 + V2 hit rate.

2. **The fixes are upstream-able.** Adding a missing `GKYL_CU_DH` to a
   kernel definition or flipping `GKYL_CU_D → GKYL_CU_DH` on a
   dispatcher is a strict improvement on the CUDA path too — same
   resolved entity, just removes the silent mismatch. PRs back to the
   gkeyll-agent-dev main branch are low-risk.

3. **Two-pass annotation audit at end of port.** When the port closes,
   one-shot grep across `core/`, `moments/`, `vlasov/`,
   `gyrokinetic/`, `pkpm/` looking for:
   - `.c` files in `ker/` whose first definition lacks `GKYL_CU_DH` while
     the matching header has it (development #1 pattern).
   - `^GKYL_CU_D$` immediately followed by `static (void|int|...) ...`
     (development #2 pattern).
   Catches latent AMD-port bugs in code paths we didn't exercise in
   this round.

---

## V2 changeset summary

```
machines/configure.frontier-gpu.sh  (V0: --app=core → --app=vlasov)
moments/Makefile-moments            (V1: 6 edits)
moments/zero/wave_geom.c            (V1: gate sweep)
moments/zero/wv_euler.c             (V1)
moments/zero/wv_maxwell.c           (V1)
moments/zero/wv_ten_moment.c        (V1)
moments/unit/ctest_wave_geom.c      (V1)
moments/unit/ctest_wv_euler.c       (V1)
moments/unit/ctest_wv_maxwell.c     (V1)
moments/unit/ctest_wv_ten_moment.c  (V1)
vlasov/Makefile-vlasov                                    (V2 planned)
vlasov/zero/{91 files: gates flipped}                     (V2 planned)
vlasov/zero/vlasov_lte_proj_on_basis_cu.cu                (V2 planned)
vlasov/ker/canonical_pb/{30 files: GKYL_CU_DH added}      (V2 unplanned #1)
vlasov/zero/gkyl_positivity_shift_vlasov_priv.h           (V2 unplanned #2)
```

### V3 — Next

Per the plan, V3 sweeps `vlasov/apps/` (2 files: `vlasov_comms.c`,
`vlasov_lw.c`) for NCCL gate widening to the Phase 6 disjunction. Both
already call `gkyl_nccl_comm_new`; the priv header's gate is correct,
but the app-side `#ifdef GKYL_HAVE_NCCL` blocks need to widen to
`defined(NCCL) || defined(RCCL)`. Then verify the Lua wrapper
(`vlasov_lw.c`) links cleanly under hipcc with the LuaJIT path that
Phase 1 set up. V3 gate: the Vlasov-app driver binary (the entry point
for the fiducial Lua sims) builds and links.

V4 is the first single-GPU fiducial run (`rt_vlasov_twostream_p2`).
