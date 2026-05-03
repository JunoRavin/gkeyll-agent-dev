// Linear-solver dispatcher header is CUDA-only per amd_port_plan.md §7.
// On the HIP path this resolves to nothing — callers in HIP TUs that try
// to use a culinsolver_* symbol get a clear undefined-reference error.
#ifdef GKYL_HAVE_CUDA

#ifdef GKYL_HAVE_CUDSS

// Code was compiled with cuDSS.
#include <gkyl_cudss_ops.h>

#else

// Code was compiled without cuDSS, use cuSolver/cuSparse instead.
#include <gkyl_cusolver_ops.h>

#endif

#endif // GKYL_HAVE_CUDA
