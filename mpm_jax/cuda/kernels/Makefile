# Build CUDA FFI kernels for JAX integration.
#
# Usage:
#   make                    # build all kernels
#   make clean              # remove .so files
#   make FFI_INC=<path>     # override XLA FFI include path
#
# Requires: nvcc, jax (for include path)

PROJECT_ROOT := $(abspath ../../..)

NVCC       := nvcc
PYTHON     ?= $(PROJECT_ROOT)/.venv/bin/python
ARCH       := sm_90
NVCC_FLAGS := -arch=$(ARCH) -O3 --use_fast_math -std=c++17 -Xcompiler -fPIC -shared
BUILD_DIR  ?= $(PROJECT_ROOT)/build/ffi

# XLA FFI headers — auto-detect from jax install
FFI_INC ?= $(shell $(PYTHON) -c 'import jax.ffi; print(jax.ffi.include_dir())' 2>/dev/null)

ifeq ($(FFI_INC),)
$(error Cannot find JAX FFI include directory. Set FFI_INC manually or ensure jax is installed.)
endif

INCLUDES := -I$(FFI_INC)

# Targets
TARGETS := \
	$(BUILD_DIR)/libp2g_scatter.so \
	$(BUILD_DIR)/libp2g_scatter_warp.so \
	$(BUILD_DIR)/libp2g_scatter_smem.so \
	$(BUILD_DIR)/libp2g_fused.so

.PHONY: all clean

all: $(TARGETS)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/libp2g_scatter.so: p2g_scatter.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/libp2g_scatter_warp.so: p2g_scatter_warp.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/libp2g_scatter_smem.so: p2g_scatter_smem.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/libp2g_fused.so: p2g_fused.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(TARGETS) *.so
