#pragma once

// GPU reduction translation shim.
//
// Provides a single include point that resolves to either CUB (CUDA) or
// hipCUB (HIP) for block-level reduction primitives used by the
// array_reduce / array_dg_reduce / array_integrate kernels.
//
// Build-flag gating (set via CFLAGS by the top-level Makefile):
//   GKYL_HAVE_HIP   -> hipCUB on rocPRIM
//   GKYL_HAVE_CUDA  -> CUB
//
// Strategy: a `namespace cub = hipcub;` alias on the HIP path lets
// existing source code continue to write `cub::BlockReduce<...>`,
// `cub::Max()`, `cub::Min()`, `cub::Sum()` unchanged. The alias is
// preferred over `#define cub hipcub` because it is scoped, type-safe,
// and plays correctly with template instantiation and ADL.

#if defined(GKYL_HAVE_HIP)

// hipCUB pulls in C++ template content; if reached transitively from
// inside an `extern "C" {}` block it must be wrapped to escape the
// surrounding C linkage. The reduction _cu.cu files include this shim
// at file scope (before their `extern "C" {}` blocks), so direct
// inclusion at file scope does not need the wrap. The wrap below is
// defensive against future call sites that include the shim from
// inside an `extern "C" {}` block.
#if defined(__cplusplus)
extern "C++" {
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
}
#else
#error "gkyl_gpu_reduce.h must be included from C++ TUs only (.cu files)"
#endif

#elif defined(GKYL_HAVE_CUDA)

#include <cub/cub.cuh>

#endif
