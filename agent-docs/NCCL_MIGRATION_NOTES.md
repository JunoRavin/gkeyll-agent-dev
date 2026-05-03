# NCCL Communicator Migration Notes

This document describes changes made to `nccl_comm.c` to address compatibility
issues across NCCL versions, particularly when migrating from NCCL 2.18 to
newer releases (2.19+). It provides NVIDIA documentation references for each
change and explains why these changes are necessary even when the original code
may appear to work on some NCCL versions.

## Summary of Changes

1. **Consolidated per-direction NCCL groups in `array_per_sync`** into a single
   `ncclGroupStart`/`ncclGroupEnd` call.
2. **Separated CUDA memory operations from NCCL group calls** — all buffer
   resizing and `copy_to_buffer` operations now occur before `ncclGroupStart`;
   all `copy_from_buffer` operations occur after `cudaStreamSynchronize`.
3. **Guarded NCCL group calls on actual operations** — ranks that enter
   `array_per_sync` but have no periodic neighbors in the periodic directions
   (e.g., interior ranks in a phase-space decomposition that touch velocity-space
   edges but not configuration-space edges) now skip `ncclGroupStart`/`ncclGroupEnd`
   entirely, avoiding empty group calls that trigger a segfault in NCCL 2.25.
4. **Added `MPI_Barrier` to the `barrier` function** — the previous
   implementation only performed a local `cudaStreamSynchronize`, which is not
   a cross-rank synchronization primitive.

## NCCL Version Context

### NCCL ≤ 2.18: Implicit Serialization

In NCCL 2.18 and earlier, the library provided implicit communicator-level
serialization. Operations submitted to a communicator were internally queued
and matched across ranks, even across separate `ncclGroupStart`/`ncclGroupEnd`
boundaries. This meant that:

- Per-direction groups (separate group start/end for each edge direction)
  worked because NCCL would internally match send/recv pairs across group
  boundaries.
- Ranks that raced ahead to the next group call would have their operations
  queued behind pending operations from other ranks.
- Empty groups (ncclGroupStart + ncclGroupEnd with no operations) were
  effectively no-ops.

This implicit serialization was an implementation detail, not part of the
API contract.

**Documentation reference (NCCL 2.18 group calls)**:
https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/usage.html

### NCCL 2.19: Nonblocking Mode and Stricter Group Semantics

NCCL 2.19 introduced `ncclCommInitRankConfig` with the `config.blocking = 0`
option for nonblocking communicators. This fundamentally changed how group
operations are processed:

- `ncclGroupEnd()` may return `ncclInProgress` instead of blocking until
  completion. Users must poll with `ncclCommGetAsyncError()` until
  `ncclSuccess`.
- CUDA stream operations (like `cudaStreamSynchronize`) must only be called
  **after** `ncclGroupEnd()` returns — not inside the group.
- Operations between `ncclGroupStart()` and `ncclGroupEnd()` may return
  without having enqueued work on the CUDA stream.
- The implicit cross-group serialization from 2.18 was removed for
  performance reasons.

**Key documentation references**:
- [Group Calls (NCCL 2.29 — current spec)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html):
  > "When called inside a group, stream operations can return without having
  > enqueued the operation on the stream, and stream operations like
  > cudaStreamSynchronize can therefore be called only after ncclGroupEnd
  > returns."
- [Creating a Communicator — nonblocking mode](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html):
  > "If the return code of any NCCL function is ncclInProgress, it means the
  > operation is in the process of being enqueued in the background, and users
  > must query the states of the communicators until all the states become
  > ncclSuccess before calling another NCCL function."
- [NCCL 2.19.3 Release Notes](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-19-3.html)

### NCCL 2.25: Empty Group Segfault

NCCL 2.25.1 contains a bug where calling `ncclGroupStart()` followed by
`ncclGroupEnd()` with **no operations** between them — while other ranks on
the same communicator have actual `ncclSend`/`ncclRecv` operations in their
group — causes a segfault inside `ncclCommSetAsyncError()` called from
`ncclGroupEndInternal()`. The crash manifests as a SIGSEGV at a very small
address (e.g., `0x78b35`), indicating the NCCL communicator handle is
corrupted.

This bug is related to a broader class of issues with nonblocking
communicator initialization in NCCL 2.21–2.25. A known issue
([NVIDIA/nccl#1605](https://github.com/NVIDIA/nccl/issues/1605)) documents
segfaults in `ncclGroupCommJoin` with nonblocking communicators, affecting
NCCL 2.21.5 and 2.25.1, with a workaround of setting
`NCCL_COMM_BLOCKING=1`.

The empty group pattern arises naturally in Gkeyll when:
- The communicator is extended to phase space (e.g., `{4, 1}` decomposition
  for 4 ranks in x, 1 in v).
- All ranks have `touches_any_edge = true` because every rank is on both
  the lower and upper edge of the velocity dimension (only 1 rank in v).
- Interior ranks (not on any configuration-space edge) enter `array_per_sync`,
  iterate over the periodic directions (only x), find no periodic neighbors,
  and reach `ncclGroupStart`/`ncclGroupEnd` with zero operations.

**Fix**: Guard the NCCL group block with `has_nccl_ops` — only call
`ncclGroupStart`/`ncclGroupEnd` if the rank actually has send/recv
operations to post.

**References**:
- [NVIDIA/nccl#1605 — segfault with nonblocking init](https://github.com/NVIDIA/nccl/issues/1605)
- [NCCL 2.25.1 documentation](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2251/user-guide/docs/index.html)

### NCCL 2.29: Apparent Tolerance of Legacy Patterns

The original `nccl_comm.c` code (with per-direction groups) runs without
error on at least one cluster using NCCL 2.29. This does not mean the code
is correct — it means NCCL 2.29 is more tolerant of certain patterns that
crash in 2.25 and that violated the API contract since 2.19. Possible
explanations:

- NCCL 2.29 may have fixed the empty-group segfault bug present in 2.25.
- NCCL 2.29 may have restored some implicit serialization for common
  patterns.
- The specific GPU interconnect topology or driver version on that cluster
  may exercise a different code path within NCCL.

Relying on this tolerance is fragile. The changes in this PR make the code
correct with respect to the documented API contract, ensuring portability
across NCCL versions.

**References**:
- [NCCL 2.29.7 documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [Release Notes index](https://docs.nvidia.com/deeplearning/nccl/release-notes/index.html)

## Detailed Change Descriptions

### 1. Consolidated Groups in `array_per_sync`

**Before**: Each periodic direction and edge side had its own
`ncclGroupStart`/`ncclGroupEnd`, with `ncclSend`/`ncclRecv` and buffer
operations interleaved inside each group.

**After**: All multi-rank periodic send/recv operations are batched into a
single `ncclGroupStart`/`ncclGroupEnd` call. Self-periodic cases (same-rank
copies) are handled before the group with local buffer copies only.

**Why**: The NCCL point-to-point documentation states:

> "If multiple ncclSend() and ncclRecv() operations need to progress
> concurrently to complete, they must be fused within a
> ncclGroupStart()/ncclGroupEnd() section."

— [Point-to-point communication (NCCL 2.29)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html)

With per-direction groups, matching operations from peer ranks on opposite
edges were split across separate groups, violating this requirement.

### 2. Separated Buffer Work from NCCL Calls

**Before**: `gkyl_array_copy_to_buffer`, `gkyl_mem_buff_resize`, and range
computations were interleaved with `ncclSend`/`ncclRecv` inside the group.

**After**: All buffer preparation (range computation, resize,
`copy_to_buffer`) happens in Phase 1 before `ncclGroupStart`. Only
`ncclRecv`/`ncclSend` calls appear between `ncclGroupStart` and
`ncclGroupEnd`. Buffer readout (`copy_from_buffer`) happens after
`cudaStreamSynchronize`.

**Why**: The group calls documentation states:

> "When called inside a group, stream operations can return without having
> enqueued the operation on the stream."

— [Group Calls (NCCL 2.29)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html)

CUDA memory operations submitted inside a group are not managed by NCCL and
may execute out of order relative to the deferred NCCL operations. Separating
them ensures correct ordering.

### 3. Guarded Group Calls on Actual Operations

**Before**: All ranks that pass the `touches_any_edge` check execute
`ncclGroupStart`/`ncclGroupEnd`, even if they have no periodic neighbors in
the periodic directions (resulting in empty groups).

**After**: A `has_nccl_ops` flag checks whether any recv or send operations
were prepared. Only ranks with actual operations enter the NCCL group.

**Why**: Empty groups trigger a segfault in NCCL 2.25.1 (see above). Even
in versions where empty groups don't crash, they are unnecessary overhead
and do not match any documented use pattern in the NCCL documentation.

### 4. Cross-Rank Barrier

**Before**: The `barrier` function only polled `ncclCommGetAsyncError` and
called `cudaStreamSynchronize` — both are local operations that do not
synchronize across ranks.

**After**: `MPI_Barrier(nccl->mcomm)` is called after the local
synchronization, ensuring all ranks have completed their pending operations
before any rank proceeds.

**Why**: Without a cross-rank barrier, interior ranks (which return early
from `array_per_sync`) can race ahead to `array_sync` and submit new NCCL
group operations on the communicator while edge ranks are still executing
their `per_sync` group. The group calls documentation notes:

> "users still need to guarantee the same issuing order of the operations
> among different GPUs."

— [Group Calls (NCCL 2.29)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html)

## Additional References

- [NCCL Documentation Archives](https://docs.nvidia.com/deeplearning/nccl/archives/index.html) —
  for comparing documentation across specific NCCL versions.
- [NCCL Point-to-Point API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html) —
  `ncclSend` and `ncclRecv` function signatures and requirements.
- [NCCL Examples](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html) —
  official examples showing correct group usage patterns.
- [NCCL GitHub Issues](https://github.com/NVIDIA/nccl/issues) —
  for tracking known bugs and workarounds.
