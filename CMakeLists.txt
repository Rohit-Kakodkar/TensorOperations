cmake_minimum_required(VERSION 3.22)
project(TensorKokkos CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TENSOR_OPS_ENABLE_TIMING "Enable timing instrumentation for evaluator operations" OFF)
option(TENSOR_OPS_BUILD_BENCHMARKS "Build performance benchmarks" OFF)
# Compile-time verification that a generic reshape reproduces its source's
# element->offset map. O(num_elements) constexpr work per instantiation, so it
# defaults to debug builds only (see TENSOR_OPS_VERIFY_RESHAPE in
# include/TensorOperations/TileLayout.hpp); this option forces it on regardless.
option(TENSOR_OPS_VERIFY_RESHAPE "Compile-time-verify generic reshape plans (implied by debug builds)" OFF)

include(FetchContent)

FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    # develop (not a tagged release): needed for kokkos/kokkos#9184, which fixes
    # an alignment bug in simd_partial_load/store used by the GEMM micro-kernel's
    # masked-remainder path. The fix is merged but downstream of the 5.1.1 tag.
    GIT_TAG        develop
)
FetchContent_MakeAvailable(kokkos)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_library(tensor INTERFACE)
target_include_directories(tensor INTERFACE include/)
target_link_libraries(tensor INTERFACE Kokkos::kokkos)
target_compile_features(tensor INTERFACE cxx_std_20)
target_compile_options(tensor INTERFACE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-fbracket-depth=2048>)

if(TENSOR_OPS_ENABLE_TIMING)
  target_compile_definitions(tensor INTERFACE TENSOR_OPS_ENABLE_TIMING)
endif()

if(TENSOR_OPS_VERIFY_RESHAPE)
  target_compile_definitions(tensor INTERFACE TENSOR_OPS_VERIFY_RESHAPE=1)
endif()

enable_testing()
include(GoogleTest)

# test_concept — no Kokkos runtime; GTest provides main()
add_executable(test_concept tests/test_concept.cpp)
target_link_libraries(test_concept PRIVATE tensor GTest::gtest_main)
gtest_discover_tests(test_concept)

# test_nodehandle — custom main() wraps Kokkos init/finalize
add_executable(test_nodehandle tests/test_nodehandle.cpp)
target_link_libraries(test_nodehandle PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_nodehandle)

# test_contraction_node — custom main() wraps Kokkos init/finalize
add_executable(test_contraction_node tests/test_contraction_node.cpp)
target_link_libraries(test_contraction_node PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_contraction_node)

# test_graph — custom main() wraps Kokkos init/finalize
add_executable(test_graph tests/test_graph.cpp)
target_link_libraries(test_graph PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_graph)

# test_device_tuple — custom main() wraps Kokkos init/finalize
add_executable(test_device_tuple tests/test_device_tuple.cpp)
target_link_libraries(test_device_tuple PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_device_tuple)

# test_combine_node — custom main() wraps Kokkos init/finalize
add_executable(test_combine_node tests/test_combine_node.cpp)
target_link_libraries(test_combine_node PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_combine_node)

# test_tiling — custom main() wraps Kokkos init/finalize
add_executable(test_tiling tests/test_tiling.cpp)
target_link_libraries(test_tiling PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_tiling)

# test_tiled_layout — custom main() wraps Kokkos init/finalize
add_executable(test_tiled_layout tests/test_tiled_layout.cpp)
target_link_libraries(test_tiled_layout PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_tiled_layout)

# test_scratch_allocator — custom main() wraps Kokkos init/finalize
add_executable(test_scratch_allocator tests/test_scratch_allocator.cpp)
target_link_libraries(test_scratch_allocator PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_scratch_allocator)
# test_tile_layout_stride — custom main() wraps Kokkos init/finalize
add_executable(test_tile_layout_stride tests/test_tile_layout_stride.cpp)
target_link_libraries(test_tile_layout_stride PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_tile_layout_stride)

# test_static_layout — custom main() wraps Kokkos init/finalize.
# Forces the reshape verification on regardless of build type: this is the file
# that covers the generic reshape, so its invariant should be checked in every
# CI configuration, not only debug ones.
add_executable(test_static_layout tests/test_static_layout.cpp)
target_link_libraries(test_static_layout PRIVATE tensor GTest::gtest)
target_compile_definitions(test_static_layout PRIVATE TENSOR_OPS_VERIFY_RESHAPE=1)
gtest_discover_tests(test_static_layout)

# test_reorder_scratch — custom main() wraps Kokkos init/finalize
add_executable(test_reorder_scratch tests/test_reorder_scratch.cpp)
target_link_libraries(test_reorder_scratch PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_reorder_scratch)

# test_slot_node — custom main() wraps Kokkos init/finalize
add_executable(test_slot_node tests/test_slot_node.cpp)
target_link_libraries(test_slot_node PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_slot_node)

# test_adopted_output — custom main() wraps Kokkos init/finalize
add_executable(test_adopted_output tests/test_adopted_output.cpp)
target_link_libraries(test_adopted_output PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_adopted_output)

# test_slot_store — custom main() wraps Kokkos init/finalize
add_executable(test_slot_store tests/test_slot_store.cpp)
target_link_libraries(test_slot_store PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_slot_store)

# test_dag_graph — custom main() wraps Kokkos init/finalize
add_executable(test_dag_graph tests/test_dag_graph.cpp)
target_link_libraries(test_dag_graph PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_dag_graph)

# test_slot_liveness — custom main() wraps Kokkos init/finalize
add_executable(test_slot_liveness tests/test_slot_liveness.cpp)
target_link_libraries(test_slot_liveness PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_slot_liveness)

# ---------------------------------------------------------------------------
# Negative-compilation tests for the slot guards.
#
# Asserting a guard predicate is false proves the predicate; it does NOT prove
# the evaluator consults it, and a guard that is defined but never read looks
# identical from inside a static_assert. These targets build a configuration the
# guard forbids and ctest asserts the BUILD fails.
#
# EXCLUDE_FROM_ALL so `make` stays green; each is built on demand by its test.
# Case 0 is the control -- the same shape without the violation -- and must
# build, so a file that breaks for unrelated reasons cannot masquerade as every
# guard passing.
#
# NOTE: unlike every other test here, these INVOKE THE COMPILER AT TEST TIME.
# On the CUDA build that means `module load cudatoolkit/<ver>` must be in effect
# when ctest runs, not merely when cmake --build ran; without it nvcc_wrapper
# fails with "nvcc: command not found" and the cases report as failures that
# have nothing to do with the guards.
# ---------------------------------------------------------------------------
foreach(case 0 1 2)
  add_executable(slot_guard_neg_${case} EXCLUDE_FROM_ALL
                 tests/negative/slot_guard_negative.cpp)
  target_link_libraries(slot_guard_neg_${case} PRIVATE tensor)
  target_compile_definitions(slot_guard_neg_${case}
                             PRIVATE SLOT_NEG_CASE=${case})
  add_test(NAME SlotGuardNegative.Case${case}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                   --target slot_guard_neg_${case} --config $<CONFIG>)
endforeach()
# Case 0 must compile. Cases 1 and 2 must fail, and must fail FOR THE RIGHT
# REASON: WILL_FAIL alone is satisfied by any nonzero exit, so an unrelated
# breakage in the file would masquerade as both guards working. Matching the
# guard's own diagnostic pins it to the intended static_assert -- in particular
# to the slot-specific message rather than the fused-operand extent one, which
# is the mistake a reader of a bare failure would most likely make.
# PASS_REGULAR_EXPRESSION supersedes the exit code, so WILL_FAIL is not also set.
set_tests_properties(SlotGuardNegative.Case1 SlotGuardNegative.Case2
                     PROPERTIES PASS_REGULAR_EXPRESSION
                     "operand [AB] is a SLOT and perm[AB] is not the identity")

# Benchmarks (opt-in: -DTENSOR_OPS_BUILD_BENCHMARKS=ON). No gtest; own main().
if(TENSOR_OPS_BUILD_BENCHMARKS)
  # KokkosKernels — needed for bench_matrix_multiply's BLAS3 GEMM path.
  # KokkosKernels calls find_package(Kokkos) internally; point it at the
  # build-tree KokkosConfig.cmake rather than a system install.
  set(Kokkos_DIR "${kokkos_BINARY_DIR}" CACHE PATH "" FORCE)
  set(KokkosKernels_ENABLE_TESTS     OFF CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_EXAMPLES  OFF CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_PERFTESTS OFF CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_DOCS      OFF CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_TPL_CUBLAS   OFF CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_TPL_CUSOLVER OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
      kokkoskernels
      GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels.git
      GIT_TAG        develop
  )
  FetchContent_MakeAvailable(kokkoskernels)

  add_executable(bench_team_contraction
    benchmarks/team_contraction/bench_team_contraction.cpp)
  target_link_libraries(bench_team_contraction PRIVATE tensor)

  add_executable(bench_sem_stiffness
    benchmarks/sem_stiffness/bench_sem_stiffness.cpp)
  target_link_libraries(bench_sem_stiffness PRIVATE tensor)

  add_executable(bench_multilevel_contraction
    benchmarks/multilevel_contraction/bench_multilevel_contraction.cpp)
  target_link_libraries(bench_multilevel_contraction PRIVATE tensor)

  add_executable(bench_permuted_contraction
    benchmarks/permuted_contraction/bench_permuted_contraction.cpp)
  target_link_libraries(bench_permuted_contraction PRIVATE tensor)

  add_executable(bench_permuted_combine
    benchmarks/permuted_combine/bench_permuted_combine.cpp)
  target_link_libraries(bench_permuted_combine PRIVATE tensor)

  add_executable(bench_matrix_multiply
    benchmarks/matrix_multiply/bench_matrix_multiply.cpp)
  target_link_libraries(bench_matrix_multiply PRIVATE tensor Kokkos::kokkoskernels)
endif()