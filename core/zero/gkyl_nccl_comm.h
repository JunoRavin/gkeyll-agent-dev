#pragma once

#include <gkyl_gpu_runtime.h>

// RCCL is API-compatible with NCCL: same `nccl*` symbol names, same enum
// values, same calling conventions. The header lives at <rccl/rccl.h>
// under ROCm. Source files that want either implementation gate on the
// disjunction below; the include switch lives here so the rest of the
// codebase keeps using the platform-agnostic `nccl*` API.
#if defined(GKYL_HAVE_NCCL) || defined(GKYL_HAVE_RCCL)

#include <mpi.h>
#if defined(GKYL_HAVE_RCCL)
#include <rccl/rccl.h>
#else
#include <nccl.h>
#endif
#include <gkyl_comm.h>
#include <gkyl_rect_decomp.h>

// input to create new NCCL communicator
struct gkyl_nccl_comm_inp {
  MPI_Comm mpi_comm; // MPI communicator to use
  const struct gkyl_rect_decomp *decomp; // pre-computed decomposition
  bool sync_corners; // should we sync corners?
  bool device_set; // Whether the device has been set already (should only happen once).
  cudaStream_t custream; // CUDA stream to use.
};

/**
 * Return a new NCCL communicator. The methods in the communicator work
 * on arrays defined on the decomp specified in the input. This
 * contract is not checked (and can't be checked) and so the user of
 * this object must ensure consistency of the arrays and the
 * decomposition.
 *
 * @param inp Input struct to use for initialization
 * @return New NCCL communicator
 */
struct gkyl_comm *gkyl_nccl_comm_new(const struct gkyl_nccl_comm_inp *inp);

#endif
