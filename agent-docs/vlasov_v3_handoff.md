# AMD Port — Vlasov V3 Handoff

V3 (Vlasov-app driver build) complete. The `gkeyll` Lua-app driver
executable is built (72 KB), embeds the correct rpath for the configured
`--app=vlasov`, and links against `libg0vlasov.so` plus the ROCm
runtime stack (`libamdhip64`, `librocblas`, `librocsolver`, `librccl`).
The Vlasov-Maxwell fiducial sims should now have an executable to run
against in V4. Continuation of [`vlasov_v0_v1_handoff.md`](vlasov_v0_v1_handoff.md)
and [`vlasov_v2_handoff.md`](vlasov_v2_handoff.md); plan in
[`amd_vlasov_port_plan.md`](amd_vlasov_port_plan.md).

This handoff focuses on the **three unplanned developments** in V3, all
sharing a common shape: code paths hardcoded to `pkpm` (the default
build-app) that nobody had ever exercised with `--app=vlasov`,
`--app=moments`, or `--app=gyrokinetic`.

---

## Planned V3 work — NCCL gate widening (4 sites)

Per the plan, V3 widens `vlasov/apps/` NCCL gates from `#ifdef
GKYL_HAVE_NCCL` to `#if defined(GKYL_HAVE_NCCL) || defined(GKYL_HAVE_RCCL)`,
matching the Phase 6 pattern that was applied to `core/zero/`. Four sites
in three files:

```
vlasov/apps/vlasov_lw.c:28      (#include guard for <gkyl_nccl_comm.h>)
vlasov/apps/vlasov_lw.c:1986    (runtime NCCL comm construction)
vlasov/apps/vlasov_comms.c:12   (#include + impl block)
vlasov/apps/gkyl_vlasov_comms.h:10  (header API guard)
```

The Phase 6 priv-header include switch in `core/zero/gkyl_nccl_comm.h`
already handles `<rccl/rccl.h>` vs `<nccl.h>` selection; the symbols are
identical (RCCL is API-compatible with NCCL — same `nccl*` names and
enum values), so no app-side translation is needed beyond the gate.

This was uneventful — single sed sweep, no surprises.

---

## Unplanned development #1: configure `${BUILD_DIR}` not escaped for non-pkpm arms

### Symptom

After running `source machines/configure.frontier-gpu.sh` (which now
passes `--app=vlasov`), `config.mak` contained:

```makefile
FIN_APP_LIB_DIR=-L..//vlasov
```

Note the **double slash and missing build-dir component**. At link time
this caused the gkeyll driver to fail to find `libg0vlasov.so` in the
expected location.

### Root cause

In `configure` lines 365, 372, 379 (the moments / vlasov / gyrokinetic
arms of the `--app=...` switch):

```sh
FIN_APP_LIB_DIR="-L../${BUILD_DIR}/vlasov"
```

The `${BUILD_DIR}` here is **shell-expanded at configure time**, but
`BUILD_DIR` is a Makefile variable that doesn't exist in the configure
script's shell environment. It expands to the empty string, leaving
the dangling `..//vlasov` path.

Compare the pkpm default at line 14, which gets it right:

```sh
FIN_APP_LIB_DIR="-L../\${BUILD_DIR}/pkpm"
```

The backslash escapes the `$`, so `${BUILD_DIR}` is written **literally**
into `config.mak`. Make then resolves it at build time, when `BUILD_DIR=hip-build`
(or `cuda-build`) has been set by the `USING_HIPCC`/`USING_NVCC` arm of
the top Makefile.

The bug existed in three of four arms — only the pkpm default was
correct. It went undetected because nobody had built a non-pkpm app on
this codebase since the configure script was last touched.

### Fix

Added the backslash escape to lines 365, 372, 379 of `configure`. CUDA
builds get the same fix transparently — they just emit `cuda-build`
instead of `hip-build` at make time.

---

## Unplanned development #2: top Makefile `gkeyll: pkpm` hardcoded

### Symptom

After fixing development #1, `make gkeyll` invoked the recursive
`make pkpm` recipe even though `BUILD_APP=vlasov`. Since pkpm has its
own GPU plumbing not yet ported to HIP (out of scope for this round),
the build either short-circuited harmlessly because pkpm was already
done from a previous cuda-build, or would have failed if attempted from
a clean tree.

### Root cause

`Makefile:456` hardcoded the dependency:

```makefile
gkeyll: pkpm ## Build Gkeyll executable
	cd gkeyll && ${MAKE} -f Makefile-gkeyll gkeyll
```

The `gkeyll` target is the same regardless of `--app=...` (it builds
the Lua-app driver against whatever `${FINAL_APP_LIB}` points at), but
its build-time dependency was hardcoded to one specific app — pkpm.

### Fix

Changed to `gkeyll: ${BUILD_APP}` (and `gkeyll-install: ${BUILD_APP}-install gkeyll`)
so the dependency resolves to the configured app. Works for any of the
four valid `--app` values.

---

## Unplanned development #3: `gkeyll/Makefile-gkeyll` rpath hardcoded to pkpm

### Symptom

After fixing #1 and #2, the gkeyll driver linked successfully but `ldd`
on the executable showed:

```
libg0vlasov.so => not found
libamdhip64.so.6 => not found
librocblas.so.4 => not found
librocsolver.so.0 => not found
```

The library was on disk at `hip-build/vlasov/libg0vlasov.so`, but the
binary's embedded `RPATH` had `./hip-build/pkpm` instead, so the dynamic
linker couldn't find it.

### Root cause

`gkeyll/Makefile-gkeyll:39`:

```makefile
LDFLAGS += -Wl,-rpath,./${BUILD_DIR}/core
LDFLAGS += -Wl,-rpath,./${BUILD_DIR}/pkpm
```

Hardcoded `pkpm` again. Same shape as #2 — the rpath depends on which
app is configured, but the rule was written for the default and never
generalized.

### Fix

```makefile
LDFLAGS += -Wl,-rpath,./${BUILD_DIR}/${BUILD_APP}
```

Verified after rebuild via `readelf -d`:

```
RPATH: [./hip-build/core:./hip-build/vlasov:./hip-build/core:./hip-build/vlasov:...]
NEEDED: libg0vlasov.so, libsci_amd, libamdhip64, librocblas, librocsolver, ...
```

(The duplicate `./hip-build/vlasov` entries come from the
`LDFLAGS += -Wl,-rpath,...` accumulation across both `LDFLAGS` and
`INSTALLED_LDFLAGS` paths in the recipe — harmless.)

---

## The pattern: pkpm-default-hardcoding

These three are different from the V2 unplanned developments. V2 was
**AMD strictness unmasking real source-level bugs nvcc tolerated**.
V3 is a different pattern: **build-system code paths hardcoded to one
specific app** (pkpm, the default) that no `--app=...` user had ever
exercised.

These bugs would manifest identically on CUDA — they're not AMD-specific.
They went undetected because:
- The default build-app is pkpm (largest dependency chain).
- Most users either build the full chain or take the default.
- The handful of users who do `--app=vlasov` had unrelated workarounds
  or never tried `make gkeyll`.

Like the V2 fixes, all three V3 fixes are upstream-able as strict
improvements to the build system. None are HIP-specific.

The remaining `--app` values (`gyrokinetic`, `moments`) inherit the
same fixes for free, since the fixes are app-agnostic.

---

## V3 changeset summary

```
configure                                    # 3 ${BUILD_DIR} escapes for non-pkpm arms
Makefile                                     # gkeyll target dep: pkpm → ${BUILD_APP}
gkeyll/Makefile-gkeyll                       # rpath: pkpm → ${BUILD_APP}
vlasov/apps/vlasov_lw.c                      # NCCL gate widening (2 sites)
vlasov/apps/vlasov_comms.c                   # NCCL gate widening
vlasov/apps/gkyl_vlasov_comms.h              # NCCL gate widening
```

### Library / binary state at end of V3

```
hip-build/core/libg0core.so         26 MB
hip-build/moments/libg0moments.so   33 MB
hip-build/vlasov/libg0vlasov.so    284 MB
hip-build/gkeyll/gkeyll             72 KB   (Lua-app driver — V3 gate met)
```

`gkeyll` `readelf -d` confirms `RPATH = ./hip-build/core:./hip-build/vlasov:...`
and `NEEDED` includes `libg0vlasov.so`, `libamdhip64`, `librocblas`,
`librocsolver`, `libmpi_amd`, `libmpi_gtl_hsa` (Cray's GPU-aware MPI
glue, which gets pulled in by mpich whether or not we use it).

### How to rebuild from scratch

```bash
cd /autofs/nccsopen-svm1_home/junoravin/working_dev/gkeyll-agent-dev
module load PrgEnv-amd rocm/6.2.4 craype-accel-amd-gfx90a cray-mpich cray-libsci
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

source machines/configure.frontier-gpu.sh   # has --app=vlasov, --use-rccl=yes
rm -rf hip-build
make moments -j 32                          # V1 gate
make vlasov  -j 32                          # V2 gate (libg0vlasov.so)
make gkeyll  -j 32                          # V3 gate (gkeyll exec)
```

---

## V4 — Next (single-GPU fiducial run)

```bash
salloc -A fus183 -N 1 --gpus 1 -t 30:00
srun -n 1 -c 8 --gpus-per-task=1 \
     hip-build/gkeyll/gkeyll vlasov/luareg/rt_vlasov_twostream_p2.lua
```

Two-stream is the simpler 1x1v fiducial — narrow kernel surface
(advection + Vlasov volume + Maxwell + LBO collisions if any), lower
poly-order arithmetic. Surfacing kernel bugs here first keeps the
search space small before V5 (`rt_vlasov_weibel_2x2v_p2`, the harder
2x2v case).

What to expect at V4:
- Sim starts up (initial-condition projection, communicator setup).
  Errors in this phase point at NCCL/RCCL bcast paths,
  `gkyl_array_reduce` host-pointer issues (Phase 4 finding pattern), or
  missing rpath / dlopen surprises.
- Sim runs (RK time stepping with Vlasov + Maxwell kernels). Errors here
  point at kernel correctness bugs — silent atomic-degradation pattern
  from Phase 5 (`__CUDA_ARCH__` alias should already protect against
  this, but verify), or kernel stack-frame overflows similar to the
  Phase 4 `ker_dev_cu_ser_2d` finding.
- Sim completes (final time reached, output written). Expected.

V4 gate: sim runs to final time without abort and produces plausible
output (e.g., total particle number conserved, total energy roughly
constant for two-stream).

V5 then runs `rt_vlasov_weibel_2x2v_p2` (2x2v EM Vlasov-Maxwell). V6 is
multi-GPU (single node, 2 ranks).

### Expected new bug categories at V4-V6

The V1-V3 hit rate suggests ~1-2 small bugs per build phase. Most likely
candidates at sim runtime:
- Host-stack-pointer-to-GPU-kernel (Phase 4 family — there are still
  `gkyl_array_reduce(&local_double, gpu_arr, OP)` patterns elsewhere
  in the codebase that we haven't audited).
- More `cu_dev_new` inner-pointer fix-ups missed (Phase 4 `tensor_field`
  family).
- More kernel-file `GKYL_CU_DH` annotation mismatches (V2 canonical_pb
  family) on rarely-exercised `vlasov/ker/` paths.

### Where to find things

- This handoff: `/autofs/nccsopen-svm1_home/junoravin/working_dev/vlasov_v3_handoff.md`
- V0/V1 handoff: [`vlasov_v0_v1_handoff.md`](vlasov_v0_v1_handoff.md)
- V2 handoff: [`vlasov_v2_handoff.md`](vlasov_v2_handoff.md)
- Vlasov port plan: [`amd_vlasov_port_plan.md`](amd_vlasov_port_plan.md)
- Core port plan: [`amd_port_plan.md`](amd_port_plan.md)
- Earlier core-port handoffs: [`phase1_handoff.md`](phase1_handoff.md)
  through [`phase6_handoff.md`](phase6_handoff.md)
- Long-form bug notes for AMD-port-relevant pitfalls in
  `~/.claude/projects/.../memory/`:
  - `array_reduce_gpu_output_pointer.md` (Phase 4)
  - `cu_dev_new_struct_inner_pointer_fixup.md` (Phase 4)
  - `cuda_arch_macro_aliasing_under_hip.md` (Phase 5)
  - `nccl_post_2_19_group_call_contract.md` (Phase 6)

### Open follow-ups (not blocking V4)

1. **`fem_poisson_bias_plane_lhs.c` annotation mismatch** (V1 sidestep,
   not real-fix). Real bug; revisit if rocSPARSE-port effort starts.
2. **Vlasov / moments unit-test correctness on GPU** (deferred from V1
   per plan). Revisit only if a fiducial sim surfaces correctness bugs.
3. **gyrokinetic / pkpm full GPU port.** Out of scope for this round.
   When picked up, the V3 build-system fixes (configure escaping,
   gkeyll dep, Makefile-gkeyll rpath) will make `--app=gyrokinetic`
   work correctly — same code paths.
4. **End-of-port two-pass annotation audit** (recommended in V2
   handoff): grep all of `core/`, `moments/`, `vlasov/`, `gyrokinetic/`,
   `pkpm/` for the V2 patterns (kernel `.c` defs lacking `GKYL_CU_DH`
   matched by header `GKYL_CU_DH` decls; `^GKYL_CU_D$` static
   dispatchers).
