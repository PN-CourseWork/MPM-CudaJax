cmake_minimum_required(VERSION 3.24)
project(mpm_cuda LANGUAGES CXX)

# CUDA arch: respect MPM_CUDA_ARCH env (e.g. sm_86 / sm_90), else autodetect.
if(DEFINED ENV{MPM_CUDA_ARCH})
    string(REPLACE "sm_" "" _arch "$ENV{MPM_CUDA_ARCH}")
    set(CMAKE_CUDA_ARCHITECTURES "${_arch}" CACHE STRING "CUDA architectures")
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()

# Skip CUDA cleanly when nvcc is absent (CPU-only installs still succeed).
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "No CUDA compiler found - skipping CUDA kernel build (JAX baseline only).")
    return()
endif()

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate the JAX FFI header from whichever Python is running scikit-build-core.
find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import jax.ffi; print(jax.ffi.include_dir())"
    OUTPUT_VARIABLE JAX_FFI_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _ffi_result
)
if(NOT _ffi_result EQUAL 0 OR JAX_FFI_INCLUDE STREQUAL "")
    message(FATAL_ERROR
        "jax.ffi.include_dir() failed. Add 'jax' to [build-system] requires "
        "or run from an env where jax is importable.")
endif()
message(STATUS "JAX FFI include: ${JAX_FFI_INCLUDE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(KERNEL_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mpm_jax/cuda/kernels)
set(KERNELS p2g_scatter p2g_scatter_warp p2g_scatter_smem p2g_fused p2g_inline p2g_v2_inline g2p_fused)

# Drop libs into mpm_jax/cuda/_lib/ in BOTH the source tree (for editable
# installs, where p2g_cuda.py looks them up via Path(__file__).parent) and
# the wheel's mpm_jax/cuda/_lib/ (for proper installs).
set(KERNEL_OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mpm_jax/cuda/_lib)

foreach(name ${KERNELS})
    add_library(${name} SHARED ${KERNEL_SRC_DIR}/${name}.cu)
    target_include_directories(${name} PRIVATE ${JAX_FFI_INCLUDE})
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-return-type>
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=940,2473>
    )
    set_target_properties(${name} PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "${name}"
        LIBRARY_OUTPUT_DIRECTORY ${KERNEL_OUT_DIR}
    )
    install(TARGETS ${name} LIBRARY DESTINATION mpm_jax/cuda/_lib)
endforeach()
