#!/bin/bash
#
# Build SuperLU and LuaJIT for the Frontier (and Frontier dev sub-system) build.
#
# LAPACK/BLAS comes from Cray LibSci (libsci_amd) — no OpenBLAS build is needed.
# cuDSS is NVIDIA-only and is skipped (the AMD path uses --use-cudss=no).
#
# Usage:  source machines/mkdeps.frontier.sh
#         (or: bash machines/mkdeps.frontier.sh, but sourcing keeps the loaded
#          modules in your shell so a follow-up configure works without
#          re-loading.)

module load PrgEnv-amd
module load rocm/6.2.4
module load craype-accel-amd-gfx90a
module load cray-mpich
module load cray-libsci

: "${PREFIX:=$HOME/gkylsoft}"
mkdir -p "$PREFIX"

cd install-deps
./mkdeps.sh \
    CC=cc CXX=CC \
    MPICC=cc MPICXX=CC \
    --build-superlu=yes \
    --build-luajit=yes \
    --prefix=$PREFIX
cd ..
