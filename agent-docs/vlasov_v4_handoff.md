# AMD Port — Vlasov V4 Handoff

V4 (single-GPU `rt_vlasov_twostream_p2` fiducial run) complete. The 1x1v
two-stream simulation runs to its final time on AMD MI250X under HIP/ROCm
6.2.4 with the V0-V3 build artifacts. Sim produces plausible output (no
abort, RK stage failures within normal range, particle/energy
conservation consistent with two-stream). Continuation of
[`vlasov_v3_handoff.md`](vlasov_v3_handoff.md); plan in
[`amd_vlasov_port_plan.md`](amd_vlasov_port_plan.md).

This handoff focuses on the **one unplanned development** in V4 — a
runtime-startup symbol-resolution failure in `core/zero/gkyl_gpu_blas.h`
— and on the **performance characterization** that surfaced at V4 and
informed the V5 investigation.

---

## Planned V4 work — fiducial run

```bash
salloc -A fus183 -N 1 --gpus 1 -t 30:00
srun -n 1 -c 8 --gpus-per-task=1 \
     hip-build/gkeyll/gkeyll vlasov/luareg/rt_vlasov_twostream_p2.lua
```

Two-stream is the simpler 1x1v fiducial: narrow kernel surface (Vlasov
streaming + acceleration + Maxwell field update + ghost-cell exchange),
poly_order=2, 64 conf cells × 32 velocity cells. Surfaces the bulk of
the per-step kernel choreography without exercising the harder 4-D phase
space paths reserved for V5.

V4 gate met: sim runs to t_end without abort and produces a plausible
final state.

---

## Unplanned development: `gkyl_mat_mm_array` runtime symbol mismatch at startup

### Symptom

After V3 closed (driver binary linked), launching the V4 fiducial
immediately died at startup with:

```
gkeyll: symbol lookup error: hip-build/vlasov/libg0vlasov.so:
        undefined symbol: _Z17gkyl_mat_mm_array...
```

Pre-V4 `ldd` showed `libg0vlasov.so` was linked correctly against the
ROCm runtime stack and `libg0core.so`. The symbol was unambiguously the
mangled C++ form (`_Z17...`) of `gkyl_mat_mm_array`, which is defined in
`core/zero/mat.c` as a plain C function (compiled by `cc`).

### Root cause

`core/zero/gkyl_gpu_blas.h` is the shim header that translates cuBLAS ↔
rocBLAS calls. C++ TUs (the `_cu.cu` files compiled under hipcc) include
this shim. Inside the shim, `gkyl_mat.h` was being included **without
an `extern "C"` wrap**:

```c
// in gkyl_gpu_blas.h (pre-V4):
#include <gkyl_mat.h>  // for enum gkyl_mat_trans
```

`gkyl_mat.h` itself has no `__cplusplus`/`extern "C"` guard. When a C++
TU pulled it in, declarations inside (including `gkyl_mat_mm_array`)
were parsed with C++ linkage and got C++ name mangling. At link/runtime
those mangled names didn't match the C-named symbols in `mat.c.o`
(compiled by `cc`), and the dynamic linker failed.

This worked fine on the CUDA path historically — nvcc handled this
slightly differently, and the mangling collision didn't surface. Under
hipcc + clang's stricter resolution (the V2 pattern of "AMD unmasking
real source-level bugs nvcc tolerated"), this latent linkage bug
manifested at V4 startup.

### Fix

Wrap the `gkyl_mat.h` include in `extern "C"` inside `gkyl_gpu_blas.h`:

```c
#ifdef __cplusplus
extern "C" {
#endif
#include <gkyl_mat.h>  // for enum gkyl_mat_trans
#ifdef __cplusplus
}
#endif
```

Symmetric improvement: the CUDA path picks up the same wrap and gets
the correct C linkage too. Eliminates the silent name-mangling
divergence between host C TUs and device-side C++ TUs that include the
shim.

### Adjacent comment correction

While editing the shim, an out-of-date comment claiming "rocblas/rocsolver
headers are extern-C-guarded and C-includable" was rewritten. Inspection
showed those headers must NOT be wrapped in `extern "C"` from Gkeyll
side: they transitively pull in `rocblas_float8.h` /
`rocblas_bfloat16.h` / `rocblas_hip_f8_impl.h`, which contain C++
templates and `operator<<` overloads that error with "templates must
have C++ linkage" if reached inside an `extern "C"` block. The function
declarations are extern-C-guarded internally already, so callers
compiled as C++ can still take their addresses with C linkage. Comment
now reflects this.

---

## Performance characterization (V4)

V4 surfaced the perf gap that V5 is investigating. Numbers below are
single-GPU MI250X (Frontier dev sub-system MI210 in our test runs)
vs. single A100 baseline the user provided.

### `rt_vlasov_twostream_p2` (1x1v, 64×32 grid, p=2)

```
Number of update calls 21259
Number of forward-Euler calls 63835
RK stage-2 failures 29 (within normal range)
Species RHS calc took 16.89 secs
Field RHS calc took 19.82 secs
Total updates took 72.6 secs
Wall: 2:05
```

Compared to A100 baseline (~3 sec total updates per the user): **~24×
slower** on this small grid. Larger 32²×16² weibel 2x2v showed a much
smaller gap (~3× — see V5 handoff), indicating the bulk of the small-grid
slowdown is launch-overhead-bound rather than compute-bound.

### Mechanism characterization

`rocprofv3 --kernel-trace` profiling at this scale revealed:

- ~6,000 kernel launches per timestep (Vlasov vol/surf + Maxwell vol/surf
  + ghost-cell exchanges + reductions + accumulations).
- AMD HIP launch overhead is ~5–15 µs per kernel vs. ~3 µs on Nvidia A100.
- The dispatching kernels (`gkyl_hyper_dg_advance_cu_kernel`,
  `gkyl_mom_calc_advance_cu_ker`) hold conservative VGPR allocations
  (128 VGPR / 1024–2060 byte scratch) due to function-pointer dispatch
  through `up->equation->vol_term` etc. — the indirect-call ABI prevents
  the LTO linker from inlining/specializing the callee.

V4 left the perf characterization at "documented and understood";
substantive optimization is out of scope for V4. The V5 handoff captures
the deeper investigation that followed when V5's fault diagnosis turned
out to share the same root with the perf gap.

---

## V4 changeset

```
core/zero/gkyl_gpu_blas.h     extern "C" wrap on <gkyl_mat.h> include;
                              comment corrected re: rocblas headers
                              (NOT C-includable — must stay outside extern "C")
```

Single file. The mechanical V0–V3 sweeps and the V2 unplanned fixes
already covered everything else V4 needed. The V4 hit rate (1 file)
matches the V1–V3 trend (~1 small bug per phase as new code-path
combinations get exercised).

---

## V5 — Next (single-GPU weibel-2x2v-p2 fiducial run)

```bash
srun -n 1 -c 8 --gpus-per-task=1 \
     hip-build/gkeyll/gkeyll vlasov/luareg/rt_vlasov_weibel_2x2v_p2.lua
```

Two-dimensional configuration space + two-dimensional velocity space
Vlasov–Maxwell. Exercises the full 2x2v p2 kernel surface that 1x1v
didn't touch — much larger generated kernels (~119-line vol kernel,
hundreds of FMA terms per `out[i]`).

What to expect at V5 (per the V0 plan and V1–V4 hit rate): possibly a
kernel-bug or runtime-resource issue specific to the bigger 2x2v p2
kernels. The V5 handoff documents what surfaced.

---

## Where to find things

- This handoff: `/autofs/nccsopen-svm1_home/junoravin/working_dev/vlasov_v4_handoff.md`
- Earlier vlasov handoffs: [`vlasov_v0_v1_handoff.md`](vlasov_v0_v1_handoff.md),
  [`vlasov_v2_handoff.md`](vlasov_v2_handoff.md),
  [`vlasov_v3_handoff.md`](vlasov_v3_handoff.md)
- Vlasov port plan: [`amd_vlasov_port_plan.md`](amd_vlasov_port_plan.md)
- Core port plan and handoffs: [`amd_port_plan.md`](amd_port_plan.md)
  through [`phase6_handoff.md`](phase6_handoff.md)
- Long-form bug notes in `~/.claude/projects/.../memory/`:
  `array_reduce_gpu_output_pointer.md`, `cu_dev_new_struct_inner_pointer_fixup.md`,
  `cuda_arch_macro_aliasing_under_hip.md`, `nccl_post_2_19_group_call_contract.md`,
  `frontier_hipcc_module_setup.md`
