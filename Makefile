# -*- makefile-gmake -*-

# Type "make help" to see help for this Makefile

# determine date of build
BUILD_DATE := $(shell date +"%Y-%m-%d %H:%M:%S %Z")
BUILD_DATE_STR := "$(BUILD_DATE)"
GIT_TIP := $(shell git describe --abbrev=12 --always --dirty=+)

# Build directory
BUILD_DIR ?= build

# Name of kernels directory
KERNELS_DIR := ker

ARCH_FLAGS ?= -march=native
CUDA_ARCH ?= 70
CFLAGS ?= -O3 -g -ffast-math -fPIC -MMD -MP -DGIT_COMMIT_ID=\"$(GIT_TIP)\" -DGKYL_BUILD_DATE=\"$(BUILD_DATE_STR)\" -DGKYL_GIT_CHANGESET=\"$(GIT_TIP)\"
LDFLAGS = 
PREFIX ?= ${HOME}/gkylsoft

FIN_APP_LIB_DIR = -L../${BUILD_DIR}/pkpm
FIN_APP_LIB = -lg0pkpm

# Include config.mak file (if it exists) to overide defaults above
-include config.mak

# Default lapack include and libraries: we prefer linking to static library
LAPACK_INC_DIR ?= $(PREFIX)/OpenBLAS/include/
LAPACK_LIB_DIR ?= $(PREFIX)/OpenBLAS/lib/
LAPACK_LIB_NAME ?= openblas
LAPACK_LIBS ?= -l${LAPACK_LIB_NAME}

HAVE_APP_FLAGS = -DGKYL_HAVE_PKPM -DGKYL_HAVE_GYROKINETIC -DGKYL_HAVE_VLASOV -DGKYL_HAVE_MOMENTS

ifeq (${BUILD_APP}, moments)
	HAVE_APP_FLAGS = -DGKYL_HAVE_MOMENTS
endif
ifeq (${BUILD_APP}, vlasov)
	HAVE_APP_FLAGS = -DGKYL_HAVE_VLASOV -DGKYL_HAVE_MOMENTS
endif
ifeq (${BUILD_APP}, gyrokinetic)
	HAVE_APP_FLAGS = -DGKYL_HAVE_GYROKINETIC -DGKYL_HAVE_VLASOV -DGKYL_HAVE_MOMENTS
endif
ifeq (${BUILD_APP}, pkpm)
	HAVE_APP_FLAGS = -DGKYL_HAVE_PKPM -DGKYL_HAVE_GYROKINETIC -DGKYL_HAVE_VLASOV -DGKYL_HAVE_MOMENTS
endif

INSTALL_PREFIX ?= ${PREFIX}
PROJ_NAME ?= gkeyll

# Determine OS we are running on
UNAME = $(shell uname)

# On OSX we should use Accelerate framework
ifeq ($(UNAME), Darwin)
	LAPACK_LIB_DIR = .
	LAPACK_INC_DIR = core # dummy
	LAPACK_LIB_NAME =
	LAPACK_LIBS = -framework Accelerate
	CFLAGS += -DGKYL_USING_FRAMEWORK_ACCELERATE
endif

# Default superlu include and libraries
SUPERLU_INC_DIR ?= $(PREFIX)/superlu/include/
ifeq ($(UNAME), Darwin)
	SUPERLU_LIB_DIR ?= $(PREFIX)/superlu/lib/
else
	SUPERLU_LIB_DIR ?= $(PREFIX)/superlu/lib64/
endif
SUPERLU_LIB_NAME ?= superlu
SUPERLU_LIBS ?= -l${SUPERLU_LIB_NAME}

# CUDA / HIP flags
#
# USING_NVCC selects the CUDA/nvcc toolchain. USING_HIPCC selects the HIP/hipcc
# toolchain. Exactly one (or neither, for CPU-only builds) is set. Both paths
# populate the unified GPU_FLAGS / GPU_LDFLAGS / GPU_LIBS variables consumed
# by the sub-makes; CUDA_LIBS / NVCC_FLAGS are kept as legacy aliases pointing
# at the unified vars so existing references keep working.
USING_NVCC =
USING_HIPCC =
NVCC_FLAGS =
GPU_FLAGS =
GPU_LDFLAGS =
GPU_LIBS =
CUDA_LIBS =
# Default SQL flags (GPU blocks below override this for GPU builds)
SQL_CFLAGS ?= -fPIC -Wno-implicit-int-float-conversion
ifneq (,$(filter $(CC),nvcc nvc))
	USING_NVCC = yes
	# On the CUDA path GPUCXX defaults to the host compiler (nvcc handles both).
	GPUCXX = $(CC)
	CFLAGS = -O3 -g --forward-unknown-to-host-compiler --use_fast_math -ffast-math -MMD -MP -fPIC -DGIT_COMMIT_ID=\"$(GIT_TIP)\" -DGKYL_BUILD_DATE=\"$(BUILD_DATE_STR)\" -DGKYL_GIT_CHANGESET=\"$(GIT_TIP)\"
	GPU_FLAGS = -x cu -dc -arch=sm_${CUDA_ARCH} -rdc=true --compiler-options="-fPIC" -Xptxas --disable-optimizer-constants
	NVCC_FLAGS = $(GPU_FLAGS)
	LDFLAGS += -arch=sm_${CUDA_ARCH} -rdc=true
	ifdef CUDAMATH_LIB_DIR
		GPU_LIBS = -L${CUDAMATH_LIB_DIR}
	endif
	GPU_LIBS += -lcublas -lcusparse -lcusolver
	CUDA_LIBS = $(GPU_LIBS)
	CFLAGS += -DGKYL_HAVE_CUDA -DGKYL_HAVE_GPU
	SQL_CFLAGS = --forward-unknown-to-host-compiler -fPIC
endif
# HIP / hipcc path
ifeq ($(GPUCXX),hipcc)
	USING_HIPCC = yes
	# Host code is built with $(CC) (e.g. Cray cc); device .cu files are built
	# with $(GPUCXX) (hipcc). ROCm headers are placed on global CFLAGS because
	# host TUs transitively include gkyl_util.h -> gkyl_gpu_runtime.h ->
	# <hip/hip_runtime.h>. -D__HIP_PLATFORM_AMD__ is required because hipcc
	# auto-injects it but the host compiler does not; without it
	# <hip/hip_vector_types.h> errors. AMD's host_defines.h then handles the
	# non-hipcc case (makes __device__ / __host__ empty), so kernel .c files
	# annotated GKYL_CU_DH compile cleanly under host cc.
	CFLAGS = -O3 -g -ffast-math -fPIC -MMD -MP -DGIT_COMMIT_ID=\"$(GIT_TIP)\" -DGKYL_BUILD_DATE=\"$(BUILD_DATE_STR)\" -DGKYL_GIT_CHANGESET=\"$(GIT_TIP)\"
	CFLAGS += -DGKYL_HAVE_HIP -DGKYL_HAVE_GPU -D__HIP_PLATFORM_AMD__=1
	ifdef ROCM_PATH
		CFLAGS += -I$(ROCM_PATH)/include
	endif
	# Device-code compile flags (hipcc). -fgpu-rdc must appear in BOTH compile
	# and link or symbol resolution will fail. -x hip forces hipcc into HIP mode
	# even for .c inputs (hipcc only auto-enables HIP mode for .cu/.cpp/.cxx);
	# the kernel-override block in core/Makefile-core compiles ker/*/*.c with
	# this flag so __device__ (and therefore GKYL_CU_DH) takes effect and the
	# kernel functions land in device bitcode.
	GPU_FLAGS = -x hip --offload-arch=$(GPU_ARCH) -fPIC -fgpu-rdc
	GPU_LDFLAGS = --hip-link -fgpu-rdc
	# Final-link libs.
	ifdef ROCM_PATH
		GPU_LIBS = -L$(ROCM_PATH)/lib
	endif
	# -lamdhip64 is the HIP runtime; rocblas + rocsolver replace cuBLAS for
	# mat.c's batched LU and gemm calls (see core/zero/gkyl_gpu_blas.h shim).
	GPU_LIBS += -lamdhip64 -lrocblas -lrocsolver
endif

CFLAGS += ${HAVE_APP_FLAGS}

# Directory for storing shared data, like ADAS reaction rates and radiation fits
GKYL_SHARE_DIR ?= "${INSTALL_PREFIX}/${PROJ_NAME}/share"
CFLAGS += -DGKYL_SHARE_DIR=\"$(GKYL_SHARE_DIR)\"

# MPI paths and flags
USING_MPI =
MPI_RPATH = 
MPI_INC_DIR = core # dummy
MPI_LIB_DIR = .
ifeq (${USE_MPI}, 1)
	USING_MPI = yes
	MPI_INC_DIR = ${CONF_MPI_INC_DIR}
	MPI_LIB_DIR = ${CONF_MPI_LIB_DIR}

ifdef USING_NVCC
	MPI_RPATH = -Xlinker "-rpath,${CONF_MPI_LIB_DIR}"
else
	MPI_RPATH = -Wl,-rpath,${CONF_MPI_LIB_DIR}
endif

	MPI_LIBS = -lmpi
	CFLAGS += -DGKYL_HAVE_MPI
endif

# Read NCCL paths and flags if needed (needs MPI and NVCC)
USING_NCCL =
NCCL_INC_DIR = zero # dummy
NCCL_LIB_DIR = .
ifeq (${USE_NCCL}, 1)
ifdef USING_MPI
ifdef USING_NVCC
	USING_NCCL = yes
	NCCL_INC_DIR = ${CONF_NCCL_INC_DIR}
	NCCL_LIB_DIR = ${CONF_NCCL_LIB_DIR}
	NCCL_LIBS = -lnccl
	CFLAGS += -DGKYL_HAVE_NCCL
endif
endif
endif

# Read RCCL paths and flags if needed (needs MPI and HIPCC).
# RCCL reuses the existing nccl_comm.c via a macro switch (see plan §4); the
# include/library names differ but the compiled object is the same one used on
# the CUDA path. RCCL_LIBS / NCCL_LIBS / NCCL_INC_DIR / NCCL_LIB_DIR are kept
# disjoint so nothing in the existing CUDA path moves.
USING_RCCL =
RCCL_INC_DIR = zero # dummy
RCCL_LIB_DIR = .
ifeq (${USE_RCCL}, 1)
ifdef USING_MPI
ifdef USING_HIPCC
	USING_RCCL = yes
	RCCL_INC_DIR = ${CONF_RCCL_INC_DIR}
	RCCL_LIB_DIR = ${CONF_RCCL_LIB_DIR}
	RCCL_LIBS = -lrccl
	CFLAGS += -DGKYL_HAVE_RCCL
endif
endif
endif

# Read CUDSS paths and flags if needed (needs MPI and NVCC)
USING_CUDSS =
CUDSS_INC_DIR = zero # dummy
CUDSS_LIB_DIR = .
CUDSS_RPATH =
ifeq (${USE_CUDSS}, 1)
ifdef USING_NVCC
	USING_CUDSS = yes
	CUDSS_INC_DIR = ${CONF_CUDSS_INC_DIR}
	CUDSS_LIB_DIR = ${CONF_CUDSS_LIB_DIR}
	CUDSS_RPATH = -Xlinker "-rpath,${CONF_CUDSS_LIB_DIR}"
	CUDSS_LIBS = -lcudss
	CFLAGS += -DGKYL_HAVE_CUDSS
endif
endif

# LUA paths and flags
USING_LUA =
LUA_RPATH = 
LUA_INC_DIR = core # dummy
LUA_LIB_DIR = .
ifeq (${USE_LUA}, 1)
	USING_LUA = yes
	LUA_INC_DIR = ${CONF_LUA_INC_DIR}
	LUA_LIB_DIR = ${CONF_LUA_LIB_DIR}

ifdef USING_NVCC
	LUA_RPATH = -Xlinker "-rpath,${CONF_LUA_LIB_DIR}"
else
	LUA_RPATH = -Wl,-rpath,${CONF_LUA_LIB_DIR}
endif

	LUA_LIBS = -l${CONF_LUA_LIB}
	CFLAGS += -DGKYL_HAVE_LUA
endif

# Build directory
ifdef USING_NVCC
	BUILD_DIR = cuda-build
endif
ifdef USING_HIPCC
	BUILD_DIR = hip-build
endif

# Command to make dir
MKDIR_P ?= mkdir -p

# At this point, export all top-level variables to sub-makes and
# recurse downwards.
#
# NOTE: We use explicit 'export' instead of .EXPORT_ALL_VARIABLES: to avoid
# the "argument list too long" error. With .EXPORT_ALL_VARIABLES:, GNU Make
# causes sub-makes to also export ALL their variables (including large lists
# like SRCS, OBJS, DEPS, and MAKEFILE_LIST which can include thousands of
# .d dependency files) to every shell command they run. This pushes the
# environment size well over the system's ARG_MAX limit (2MB on Linux),
# causing execve() to fail with E2BIG even for trivial commands like mkdir.
# By explicitly exporting only the variables sub-makes actually need, we keep
# the environment small and avoid this issue.

export CC GPUCXX CFLAGS ARCH_FLAGS CUDA_ARCH GPU_ARCH LDFLAGS BUILD_DIR KERNELS_DIR
export PREFIX INSTALL_PREFIX PROJ_NAME UNAME
export USING_NVCC NVCC_FLAGS CUDA_LIBS SQL_CFLAGS CUDAMATH_LIBDIR
export USING_HIPCC GPU_FLAGS GPU_LDFLAGS GPU_LIBS ROCM_PATH
export USING_MPI MPI_INC_DIR MPI_LIB_DIR MPI_LIBS MPI_RPATH
export USING_NCCL NCCL_INC_DIR NCCL_LIB_DIR NCCL_LIBS
export USING_RCCL RCCL_INC_DIR RCCL_LIB_DIR RCCL_LIBS
export USING_CUDSS CUDSS_INC_DIR CUDSS_LIB_DIR CUDSS_LIBS CUDSS_RPATH
export USING_LUA LUA_INC_DIR LUA_LIB_DIR LUA_LIBS LUA_RPATH
export LAPACK_INC_DIR LAPACK_LIB_DIR LAPACK_LIBS LAPACK_LIB_NAME
export SUPERLU_INC_DIR SUPERLU_LIB_DIR SUPERLU_LIBS SUPERLU_LIB_NAME
export FIN_APP_LIB_DIR FIN_APP_LIB HAVE_APP_FLAGS
export MKDIR_P GKYL_SHARE_DIR BUILD_APP
export GKEYLL_SHARE_INSTALL_PREFIX SED_REPS_STR1 SED_REPS_STR2 MAKEFILE_FOR_EXT_C_INP_PHONY
export CONF_MPI_INC_DIR CONF_MPI_LIB_DIR
export CONF_NCCL_INC_DIR CONF_NCCL_LIB_DIR
export CONF_CUDSS_INC_DIR CONF_CUDSS_LIB_DIR
export CONF_LUA_INC_DIR CONF_LUA_LIB_DIR CONF_LUA_LIB

# Regression tests
${BUILD_DIR}/core/creg/%: core/creg/%.c ${BUILD_DIR}/core/libg0core.so
	cd core && $(MAKE) -f Makefile-core ../$@

${BUILD_DIR}/moments/creg/%: moments/creg/%.c ${BUILD_DIR}/moments/libg0moments.so
	cd moments && $(MAKE) -f Makefile-moments ../$@

${BUILD_DIR}/vlasov/creg/%: vlasov/creg/%.c ${BUILD_DIR}/vlasov/libg0vlasov.so
	cd vlasov && $(MAKE) -f Makefile-vlasov ../$@

${BUILD_DIR}/gyrokinetic/creg/%: gyrokinetic/creg/%.c ${BUILD_DIR}/gyrokinetic/libg0gyrokinetic.so
	cd gyrokinetic && $(MAKE) -f Makefile-gyrokinetic ../$@

${BUILD_DIR}/pkpm/creg/%: pkpm/creg/%.c ${BUILD_DIR}/pkpm/libg0pkpm.so
	cd pkpm && $(MAKE) -f Makefile-pkpm ../$@

# Unit tests
${BUILD_DIR}/core/unit/%: core/unit/%.c ${BUILD_DIR}/core/libg0core.so
	cd core && $(MAKE) -f Makefile-core ../$@

${BUILD_DIR}/moments/unit/%: moments/unit/%.c ${BUILD_DIR}/moments/libg0moments.so
	cd moments && $(MAKE) -f Makefile-moments ../$@

${BUILD_DIR}/vlasov/unit/%: vlasov/unit/%.c ${BUILD_DIR}/vlasov/libg0vlasov.so
	cd vlasov && $(MAKE) -f Makefile-vlasov ../$@

${BUILD_DIR}/gyrokinetic/unit/%: gyrokinetic/unit/%.c ${BUILD_DIR}/gyrokinetic/libg0gyrokinetic.so
	cd gyrokinetic && $(MAKE) -f Makefile-gyrokinetic ../$@

${BUILD_DIR}/pkpm/unit/%: pkpm/unit/%.c ${BUILD_DIR}/pkpm/libg0pkpm.so
	cd pkpm && $(MAKE) -f Makefile-pkpm ../$@

# Declare sub-directories as phony targets
.PHONY: core moments vlasov gyrokinetic pkpm

# sed argument to replace path in Makefile for C input files outside gkeyll/.
GKEYLL_SHARE_INSTALL_PREFIX=${INSTALL_PREFIX}/${PROJ_NAME}/share
SED_REPS_STR1=s,GKEYLL_SHARE_INSTALL_PREFIX_TAG,${GKEYLL_SHARE_INSTALL_PREFIX},g
MAKEFILE_FOR_EXT_C_INP_PHONY=
ifeq (${BUILD_APP}, moments)
	MAKEFILE_FOR_EXT_C_INP_PHONY = moments moments-unit moments-regression
endif
ifeq (${BUILD_APP}, vlasov)
	MAKEFILE_FOR_EXT_C_INP_PHONY = vlasov vlasov-unit vlasov-regression
endif
ifeq (${BUILD_APP}, gyrokinetic)
	MAKEFILE_FOR_EXT_C_INP_PHONY = gyrokinetic gyrokinetic-unit gyrokinetic-regression
endif
ifeq (${BUILD_APP}, pkpm)
	MAKEFILE_FOR_EXT_C_INP_PHONY = pkpm pkpm-unit pkpm-regression
endif
SED_REPS_STR2=s,MAKEFILE_FOR_EXT_C_INP_PHONY_TAG,${MAKEFILE_FOR_EXT_C_INP_PHONY},g


everything: regression unit gkeyll ## Build everything, including unit, regression and gkeyll exectuable

## Core infrastructure targets
core:  ## Build core infrastructure code
	cd core && $(MAKE) -f Makefile-core

core-unit: ## Build core unit tests
	cd core && $(MAKE) -f Makefile-core unit

core-regression: ## Build core regression tests
	cd core && $(MAKE) -f Makefile-core regression

core-install: ## Install core infrastructure code
	cd core && $(MAKE) -f Makefile-core install
	test -e config.mak && cp -f config.mak ${INSTALL_PREFIX}/${PROJ_NAME}/share/config.mak || echo "No config.mak"
	sed '${SED_REPS_STR1};${SED_REPS_STR2}' Makefile_for_ext_C_input > ${INSTALL_PREFIX}/${PROJ_NAME}/share/Makefile

core-clean: ## Clean core infrastructure code
	cd core && $(MAKE) -f Makefile-core clean

core-check: core ## Run unit tests in core
	cd core && $(MAKE) -f Makefile-core check

core-valcheck: core ## Run valgrind on unit tests in core
	cd core && $(MAKE) -f Makefile-core valcheck

## Moments infrastructure targets
moments: core  ## Build moments infrastructure code
	cd moments && $(MAKE) -f Makefile-moments

moments-unit: moments core-unit ## Build moments unit tests
	cd moments && $(MAKE) -f Makefile-moments unit

moments-regression: moments core-regression ## Build moments regression tests
	cd moments && $(MAKE) -f Makefile-moments regression

moments-amr-regression: moments ## Build moments AMR regression tests
	cd moments && $(MAKE) -f Makefile-moments amr_regression

moments-install: core-install ## Install moments infrastructure code
	cd moments && $(MAKE) -f Makefile-moments install
	cp -f moments/creg/rt_arg_parse.h ${INSTALL_PREFIX}/${PROJ_NAME}/include/rt_arg_parse.h

moments-clean: ## Clean moments infrastructure code
	cd moments && $(MAKE) -f Makefile-moments clean

moments-check: moments ## Run unit tests in moments
	cd moments && $(MAKE) -f Makefile-moments check

moments-valcheck: moments ## Run valgrind on unit tests in moments
	cd moments && $(MAKE) -f Makefile-moments valcheck

## Vlasov infrastructure targets
vlasov: moments  ## Build Vlasov infrastructure code
	cd vlasov && $(MAKE) -f Makefile-vlasov

vlasov-unit: vlasov moments-unit ## Build Vlasov unit tests
	cd vlasov && $(MAKE) -f Makefile-vlasov unit

vlasov-regression: vlasov moments-regression ## Build Vlasov regression tests
	cd vlasov && $(MAKE) -f Makefile-vlasov regression

vlasov-install: moments-install ## Install Vlasov infrastructure code
	cd vlasov && $(MAKE) -f Makefile-vlasov install
	cp -f vlasov/creg/rt_arg_parse.h ${INSTALL_PREFIX}/${PROJ_NAME}/include/rt_arg_parse.h

vlasov-clean: ## Clean Vlasov infrastructure code
	cd vlasov && $(MAKE) -f Makefile-vlasov clean

vlasov-check: vlasov ## Run unit tests in Vlasov
	cd vlasov && $(MAKE) -f Makefile-vlasov check

vlasov-valcheck: vlasov ## Run valgrind on unit tests in Vlasov
	cd vlasov && $(MAKE) -f Makefile-vlasov valcheck

## Gyrokinetic infrastructure targets
gyrokinetic: vlasov  ## Build Gyrokinetic infrastructure code
	cd gyrokinetic && $(MAKE) -f Makefile-gyrokinetic

gyrokinetic-unit: gyrokinetic vlasov-unit ## Build Gyrokinetic unit tests
	cd gyrokinetic && $(MAKE) -f Makefile-gyrokinetic unit

gyrokinetic-regression: gyrokinetic vlasov-regression ## Build Gyrokinetic regression tests
	cd gyrokinetic && $(MAKE) -f Makefile-gyrokinetic regression

gyrokinetic-install: vlasov-install ## Install Gyrokinetic infrastructure code
	cd gyrokinetic && $(MAKE) -f Makefile-gyrokinetic install
	cp -f gyrokinetic/creg/rt_arg_parse.h ${INSTALL_PREFIX}/${PROJ_NAME}/include/rt_arg_parse.h

gyrokinetic-clean: ## Clean Gyrokinetic infrastructure code
	cd gyrokinetic && $(MAKE) -f Makefile-gyrokinetic clean

gyrokinetic-check: gyrokinetic ## Run unit tests in Gyrokinetics
	cd gyrokinetic && $(MAKE) -f Makefile-gyrokinetic check

gyrokinetic-valcheck: gyrokinetic ## Run valgrind on unit tests in Gyrokinetics
	cd gyrokinetic && $(MAKE) -f Makefile-gyrokinetic valcheck

## PKPM infrastructure targets
pkpm: gyrokinetic  ## Build PKPM infrastructure code
	cd pkpm && $(MAKE) -f Makefile-pkpm

pkpm-unit: pkpm gyrokinetic-unit ## Build PKPM unit tests
	cd pkpm && $(MAKE) -f Makefile-pkpm unit

pkpm-regression: pkpm gyrokinetic-regression ## Build PKM regression tests
	cd pkpm && $(MAKE) -f Makefile-pkpm regression

pkpm-install: gyrokinetic-install ## Install PKPM infrastructure code
	cd pkpm && $(MAKE) -f Makefile-pkpm install
	cp -f pkpm/creg/rt_arg_parse.h ${INSTALL_PREFIX}/${PROJ_NAME}/include/rt_arg_parse.h

pkpm-clean: ## Clean PKPM infrastructure code
	cd pkpm && $(MAKE) -f Makefile-pkpm clean

pkpm-check: pkpm ## Run unit tests in PKPM
	cd pkpm && $(MAKE) -f Makefile-pkpm check

pkpm-valcheck: pkpm ## Run valgrind on unit tests in PKPM
	cd pkpm && $(MAKE) -f Makefile-pkpm valcheck

## Top-level Gkeyll target
gkeyll: pkpm ## Build Gkeyll executable
	cd gkeyll && ${MAKE} -f Makefile-gkeyll gkeyll

gkeyll-install: pkpm-install gkeyll ## Install Gkeyll executable
	cd gkeyll && ${MAKE} -f Makefile-gkeyll install

## Targets to build things all parts of the code

# build all unit tests 
unit: pkpm-unit gyrokinetic-unit vlasov-unit moments-unit core-unit ## Build all unit tests

# build all regression tests 
regression: pkpm-regression gyrokinetic-regression vlasov-regression moments-regression core-regression ## Build all regression tests

# Clean everything
clean:
	rm -rf ${BUILD_DIR}

# Check everything
check: core-check moments-check vlasov-check gyrokinetic-check pkpm-check ## Run all unit tests

# From: https://www.client9.com/self-documenting-makefiles/
.PHONY: help
help: ## Show help
	@echo "Gkeyll Makefile help. You can set parameters on the command line:"
	@echo ""
	@echo "make CC=cc -j"
	@echo ""
	@echo "Or run the configure script to set various parameters. Usually"
	@echo "defaults are all you need, specially if the dependencies are in"
	@echo "${HOME}/gkylsoft and you are using standard compilers (not building on GPUs)."
	@echo ""
	@echo "See ./configure --help for usage of configure script."
	@echo ""
	@echo "You can build only portions of the code using the specific targers below."
	@echo "Typing \"make everything\" will build the complete code"
	@echo ""
	@awk -F ':|##' '/^[^\t].+?:.*?##/ {\
        printf "\033[36m%-30s\033[0m %s\n", $$1, $$NF \
        }' $(MAKEFILE_LIST)
