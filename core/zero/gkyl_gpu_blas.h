#pragma once

// Translation shim for batched BLAS / batched LU between cuBLAS (CUDA)
// and rocBLAS + rocSOLVER (HIP). Sole include point for these libraries
// in mat.c and gkyl_mat_priv.h.
//
// Most cuBLAS calls in mat.c map onto rocBLAS by direct symbol rename
// (signatures match): handle create/destroy, dgemm, dgemm_strided_batched.
// The exceptions:
//
//   * Batched LU (cublasDgetrfBatched, cublasDgetrsBatched). These live
//     in rocSOLVER on AMD and have a different argument list — rocSOLVER
//     takes (m, n) instead of just (n) for dgetrf, and uses a strided
//     pivot layout + omits the per-batch info pointer in dgetrs. Static
//     inline wrappers translate the calling convention so callers keep
//     reading as cuBLAS.
//
//   * Operation enum (CUBLAS_OP_N etc.). Pre-port mat.c relied on the
//     numeric values matching gkyl_mat_trans (0/1/2). That's a happy
//     accident on cuBLAS but rocblas_operation uses 111/112/113. To make
//     the source uniform across platforms, all callers must funnel
//     through gkyl_mat_to_blas_op(), and never cast a gkyl_mat_trans
//     directly.
//
// rocblas/rocsolver headers are extern "C"-guarded and C-includable on
// ROCm 6.2.4 (verified). No extern "C++" wrap required here, unlike
// gkyl_gpu_runtime.h which has to escape libstdc++ leakage from
// <hip/hip_runtime.h>.

#include <gkyl_mat.h> // for enum gkyl_mat_trans

#ifdef GKYL_HAVE_HIP

#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

// --- Type renames --------------------------------------------------------
typedef rocblas_handle    cublasHandle_t;
typedef rocblas_status    cublasStatus_t;
typedef rocblas_operation cublasOperation_t;

// --- Constants -----------------------------------------------------------
#define CUBLAS_STATUS_SUCCESS rocblas_status_success
#define CUBLAS_OP_N           rocblas_operation_none
#define CUBLAS_OP_T           rocblas_operation_transpose
#define CUBLAS_OP_C           rocblas_operation_conjugate_transpose

// --- Direct symbol renames (signatures match) ---------------------------
#define cublasCreate_v2            rocblas_create_handle
#define cublasDestroy              rocblas_destroy_handle
#define cublasDgemm                rocblas_dgemm
#define cublasDgemmStridedBatched  rocblas_dgemm_strided_batched

// --- Wrappers (signatures differ) ---------------------------------------
//
// cuBLAS    cublasDgetrfBatched(h,    n,    A[], lda, pivots, info, bs)
// rocSOLVER rocsolver_dgetrf_batched(h, m, n, A[], lda, ipiv, strideP, info, bs)
//
// We only do square LU (m == n). The pivot stride is n, matching the
// contiguous (num*n)-int pivot buffer mat.c allocates.
static inline rocblas_status
gkyl_shim_cublasDgetrfBatched(rocblas_handle h, int n, double *A[], int lda,
    int *pivots, int *info, int batchSize)
{
  return rocsolver_dgetrf_batched(h, (rocblas_int)n, (rocblas_int)n,
      A, (rocblas_int)lda, pivots, (rocblas_stride)n,
      info, (rocblas_int)batchSize);
}
#define cublasDgetrfBatched gkyl_shim_cublasDgetrfBatched

// cuBLAS    cublasDgetrsBatched(h, trans, n, nrhs, A[], lda, ipiv, B[], ldb, &info, bs)
// rocSOLVER rocsolver_dgetrs_batched(h, trans, n, nrhs, A[], lda, ipiv, strideP, B[], ldb, bs)
//
// rocSOLVER returns a single status and has no per-batch info pointer.
// We surface 0/non-zero through *info_out so the cuBLAS calling
// convention at the call site is preserved.
static inline rocblas_status
gkyl_shim_cublasDgetrsBatched(rocblas_handle h, rocblas_operation trans,
    int n, int nrhs, const double *const A[], int lda, const int *ipiv,
    double *B[], int ldb, int *info_out, int batchSize)
{
  rocblas_status s = rocsolver_dgetrs_batched(h, trans,
      (rocblas_int)n, (rocblas_int)nrhs,
      (double *const *)A, (rocblas_int)lda, ipiv, (rocblas_stride)n,
      B, (rocblas_int)ldb, (rocblas_int)batchSize);
  if (info_out) *info_out = (s == rocblas_status_success) ? 0 : (int)s;
  return s;
}
#define cublasDgetrsBatched gkyl_shim_cublasDgetrsBatched

#elif defined(GKYL_HAVE_CUDA)

#include <cublas_v2.h>

#endif

// --- Common helper -------------------------------------------------------
//
// Pre-port mat.c relied on gkyl_mat_trans values numerically equalling
// cublasOperation_t (both 0/1/2). True on CUDA, false on HIP (rocblas
// uses 111/112/113). All gkyl_mat_trans → BLAS op conversions go through
// here so the source is uniform.
#if defined(GKYL_HAVE_HIP) || defined(GKYL_HAVE_CUDA)
static inline cublasOperation_t
gkyl_mat_to_blas_op(enum gkyl_mat_trans t)
{
  switch (t) {
    case GKYL_NO_TRANS:   return CUBLAS_OP_N;
    case GKYL_TRANS:      return CUBLAS_OP_T;
    case GKYL_CONJ_TRANS: return CUBLAS_OP_C;
    default:              return CUBLAS_OP_N;
  }
}
#endif
