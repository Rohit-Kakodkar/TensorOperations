#pragma once

// TENSOR_PRAGMA_UNROLL — full unroll hint for the immediately following loop.
// Each branch emits the compiler-native pragma; the fallback is a no-op.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
// NVCC/CUDA and AMD HIP/ROCm both accept the plain unroll pragma
#define TENSOR_PRAGMA_UNROLL _Pragma("unroll")
#elif defined(__INTEL_LLVM_COMPILER)
// Intel oneAPI DPC++/icx (LLVM-based)
#define TENSOR_PRAGMA_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
// GCC (and Clang, which also defines __GNUC__)
#define TENSOR_PRAGMA_UNROLL _Pragma("GCC unroll 16")
#else
#define TENSOR_PRAGMA_UNROLL
#endif
