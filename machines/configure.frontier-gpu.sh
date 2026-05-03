#!/bin/bash
#
# Configure script for Frontier (and the Frontier dev sub-system) AMD GPU build.
#
# Mirrors machines/configure.perlmutter.gpu.sh but targets HIP/ROCm via hipcc
# instead of nvcc, and links LAPACK/BLAS via Cray LibSci (libsci_amd).
#
# Usage:  source machines/configure.frontier-gpu.sh
#
# Notes:
# - Phase 1 brings up the HIP build system (configure flags, USING_HIPCC block,
#   shim header) but does NOT enable .cu compilation under hipcc yet, so the
#   resulting libg0core.so contains only the .c sources. Phases 2-6 incrementally
#   enable kernel files, reductions, rocBLAS, and RCCL.
# - --use-rccl is intentionally OFF in Phase 1: nccl_comm.c is still gated on
#   GKYL_HAVE_NCCL only. Phase 6 widens that gate and flips this flag to yes.

module load PrgEnv-amd
module load rocm/6.2.4
module load craype-accel-amd-gfx90a
module load cray-mpich
module load cray-libsci

# GPU-aware MPI for Cray MPICH. Only needed when MPI_Send/MPI_Recv is called
# directly on a GPU device pointer (i.e. gkyl_mpi_comm.c's GPU path or future
# multi-process tests that route device buffers through MPICH). The
# single-process unit-test suite and RCCL collectives do NOT need it —
# verified end of Phase 5 (see phase5_handoff.md). Set it yourself if you
# exercise that code path:
#   export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
# Disable the rocFFT runtime kernel cache (avoids contention on shared FS).
export ROCFFT_RTC_CACHE_PATH=/dev/null

: "${PREFIX:=$HOME/gkylsoft}"

./configure CC=cc GPUCXX=hipcc GPU_ARCH=gfx90a \
    ROCM_PATH=$ROCM_PATH \
    --prefix=$PREFIX \
    --app=core \
    --use-hip=yes --use-mpi=yes \
    --use-nccl=no --use-rccl=yes \
    --rccl-inc=$ROCM_PATH/include \
    --rccl-lib=$ROCM_PATH/lib \
    --use-cudss=no \
    --lapack-lib-name=sci_amd \
    --lapack-inc=$CRAY_LIBSCI_PREFIX/include \
    --lapack-lib=$CRAY_LIBSCI_PREFIX/lib \
    --mpi-inc=$CRAY_MPICH_DIR/include \
    --mpi-lib=$CRAY_MPICH_DIR/lib
