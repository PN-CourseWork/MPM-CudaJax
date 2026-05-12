# MPM-CudaJax project Makefile
#
# Usage:
#   make setup    # load modules + install deps + compile CUDA kernels
#   make cuda     # compile CUDA kernels only
#   make test     # run tests
#   make sweep    # run baseline scaling sweep
#   make clean    # remove compiled kernels + hydra/wandb output

SHELL := /bin/bash

# DTU HPC modules
MODULES := nvhpc/26.1 gcc/15.2

# Load modules in a subshell (module is a shell function, not a binary)
LOAD_MODULES := source /etc/profile.d/modules.sh 2>/dev/null; \
                for m in $(MODULES); do module load $$m 2>/dev/null; done

.PHONY: setup cuda test sweep sweep-quick sweep-all clean

setup: cuda
	uv sync --extra jax-cuda

cuda:
	@$(LOAD_MODULES); \
	cd mpm_jax/cuda/kernels && \
	uv run --extra jax-cuda make

test:
	uv run --extra jax --with pytest python -m pytest tests/ -v

sweep:
	@$(LOAD_MODULES); \
	uv run --extra jax-cuda python simulate.py -cn sweep_baseline

sweep-quick:
	@$(LOAD_MODULES); \
	uv run --extra jax-cuda python simulate.py -cn sweep_quick

sweep-all:
	@$(LOAD_MODULES); \
	uv run --extra jax-cuda python simulate.py -cn sweep_all

clean:
	rm -f mpm_jax/cuda/kernels/*.so
	rm -rf build/ffi
	rm -rf multirun/ outputs/ wandb/
	rm -f *.nsys-rep *.sqlite
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
