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

add_executable(test_team2 tests/test_team2.cpp)
target_link_libraries(test_team2 PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_team2)

add_executable(test_regroup_view tests/test_regroup_view.cpp)
target_link_libraries(test_regroup_view PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_regroup_view)

# test_slice — custom main() wraps Kokkos init/finalize
add_executable(test_slice tests/test_slice.cpp)
target_link_libraries(test_slice PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_slice)

add_executable(test_labelled_operand tests/test_labelled_operand.cpp)
target_link_libraries(test_labelled_operand PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_labelled_operand)

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

# test_label_tiles — one tile extent per label; the equivalence asserts that
# justify replacing per-node tile derivation with a graph-wide lookup.
add_executable(test_label_tiles tests/test_label_tiles.cpp)
target_link_libraries(test_label_tiles PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_label_tiles)

# test_level_liveness — pooling the level graph's slot store by live range.
# Custom main() wraps Kokkos init/finalize, as test_slot_liveness does.
add_executable(test_level_liveness tests/test_level_liveness.cpp)
target_link_libraries(test_level_liveness PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_level_liveness)

# test_level_graph — LevelPlan slot algebra + LevelGraph runtime equivalence.
# Custom main() wraps Kokkos init/finalize for the runtime half.
add_executable(test_level_graph tests/test_level_graph.cpp)
target_link_libraries(test_level_graph PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_level_graph)

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

# ---------------------------------------------------------------------------
# Negative-compilation tests for the LABEL-TILE MAP guards. Same rationale and
# same caveats as the slot guards below.
#
# Both guards protect against a silently wrong tile rather than a crash, so each
# case is pinned to its OWN diagnostic: a duplicate label and a missing label
# are independent defects and neither should trip the other's assert.
# ---------------------------------------------------------------------------
foreach(case 0 1 2)
  add_executable(labeltiles_guard_neg_${case} EXCLUDE_FROM_ALL
                 tests/negative/labeltiles_guard_negative.cpp)
  target_link_libraries(labeltiles_guard_neg_${case} PRIVATE tensor)
  target_compile_definitions(labeltiles_guard_neg_${case}
                             PRIVATE LABELTILES_NEG_CASE=${case})
  add_test(NAME LabelTilesGuardNegative.Case${case}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                   --target labeltiles_guard_neg_${case} --config $<CONFIG>)
endforeach()
set_tests_properties(LabelTilesGuardNegative.Case1 PROPERTIES
                     PASS_REGULAR_EXPRESSION
                     "may appear at most once in the map")
set_tests_properties(LabelTilesGuardNegative.Case2 PROPERTIES
                     PASS_REGULAR_EXPRESSION
                     "must appear in the graph's tile")

# ---------------------------------------------------------------------------
# Negative-compilation tests for the ARENA guard. Same rationale and same
# caveats as the slot guards above.
#
# One guard, so one failing case: the arena builds every slot at a compile-time
# offset and therefore cannot size a tile whose extents are runtime values.
# ---------------------------------------------------------------------------
foreach(case 0 1)
  add_executable(arena_guard_neg_${case} EXCLUDE_FROM_ALL
                 tests/negative/arena_guard_negative.cpp)
  target_link_libraries(arena_guard_neg_${case} PRIVATE tensor)
  target_compile_definitions(arena_guard_neg_${case}
                             PRIVATE ARENA_NEG_CASE=${case})
  add_test(NAME ArenaGuardNegative.Case${case}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                   --target arena_guard_neg_${case} --config $<CONFIG>)
endforeach()
# Pinned to the guard's own wording rather than to WILL_FAIL: a DynamicTile also
# fails LATER, on a missing num_elements, and matching that instead would let
# the guard be deleted without the test noticing.
set_tests_properties(ArenaGuardNegative.Case1 PROPERTIES
                     PASS_REGULAR_EXPRESSION
                     "slot tiles must have compile-time extents")

# ---------------------------------------------------------------------------
# Negative-compilation tests for the LEVEL guards. Same rationale and same
# caveats as the slot guards above.
#
# Each case is pinned to its OWN guard's diagnostic rather than to a shared
# one. The predicates are written so that exactly one guard fires per defect --
# the iteration-space check returns true for a level that is not homogeneous,
# leaving that defect to the homogeneity guard alone -- so a case matching two
# messages means a predicate started over-reporting.
# ---------------------------------------------------------------------------
foreach(case 0 1 2 3 4)
  add_executable(level_guard_neg_${case} EXCLUDE_FROM_ALL
                 tests/negative/level_guard_negative.cpp)
  target_link_libraries(level_guard_neg_${case} PRIVATE tensor)
  target_compile_definitions(level_guard_neg_${case}
                             PRIVATE LEVEL_NEG_CASE=${case})
  add_test(NAME LevelGuardNegative.Case${case}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                   --target level_guard_neg_${case} --config $<CONFIG>)
endforeach()
set_tests_properties(LevelGuardNegative.Case1 PROPERTIES
                     PASS_REGULAR_EXPRESSION
                     "a level must be NON-EMPTY and HOMOGENEOUS")
set_tests_properties(LevelGuardNegative.Case2 LevelGuardNegative.Case3
                     PROPERTIES PASS_REGULAR_EXPRESSION
                     "must share ONE iteration space")
set_tests_properties(LevelGuardNegative.Case4 PROPERTIES
                     PASS_REGULAR_EXPRESSION
                     "may not read a slot ITS OWN LEVEL")

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

  # --- 3D SEM stiffness, NGLL=5 --------------------------------------------
  # The order is fixed at 5 (production NGLL) in the source, not a build
  # option: it sets every tile extent in the graph AND the register blocking
  # below, and the two have to agree.
  add_executable(bench_sem_stiffness_3d
    benchmarks/sem_stiffness_3d/bench_sem_stiffness_3d.cpp)
  target_link_libraries(bench_sem_stiffness_3d PRIVATE tensor)

  # THE REGISTER BLOCKING IS PART OF THIS TARGET, NOT A TUNING PREFERENCE.
  #
  # The team GEMM register-blocks by (MT, NT, NR) and has NO REMAINDER PATH:
  # validate_gemm_tile_dims requires each staged extent to be a multiple of its
  # factor. This kernel stages SA = SK = N = 5 and SB = TE*N*N.
  #
  # 5 is prime, so the library defaults (MT=4, NT=2) divide neither SA nor SK
  # and the benchmark DOES NOT COMPILE without these overrides. 5/5/2 is the
  # measured best of the legal blockings (one 3D gradient, H100, E=262144,
  # TE=16: 5/5/2 = 1792 GF/s, 5/1/2 = 1784, 1/5/2 = 1747, 5/5/4 = 1565), so the
  # spread is small and there is no cliff being tuned around.
  #
  # Read that as the finding it is: as shipped, this library cannot express a
  # production-order SEM kernel without retuning its GEMM per target. The
  # general fix is a GEMM remainder path, which does not exist yet.
  #
  # NR=2 is also what forbids TE=1 in the benchmark's sweep -- SB = 25 is odd.
  target_compile_definitions(bench_sem_stiffness_3d PRIVATE
    TENSOR_GEMM_MT=5 TENSOR_GEMM_NT=5 TENSOR_GEMM_NR=2)

  # --- 3D SEM stiffness on the LEVEL GRAPH (Sprint 5) -----------------------
  # Note what is ABSENT, and that it is the point: no TENSOR_GEMM_* overrides.
  # The Tag2 contraction is a plain dot product with no register blocking, so
  # NGLL=5 needs no per-target GEMM retuning and TE=1 becomes legal -- neither
  # is true of bench_sem_stiffness_3d above, whose target must pin MT/NT/NR and
  # therefore cannot express TE=1 at all. TE=1 is where both this benchmark's
  # implementations are fastest.
  #
  # The sweep stops at TE=8: LevelGraph carves 35 slots with no liveness pooling
  # (that is Sprint 4), so scratch is 200 + 16500*TE bytes and TE=16 (264 KB)
  # exceeds the H100's ~227 KB opt-in limit. The level-fusion spike's best tile
  # does not survive four unpooled levels.
  add_executable(bench_sem3d_levelgraph
    benchmarks/sem_stiffness_3d/bench_sem3d_levelgraph.cpp)
  target_link_libraries(bench_sem3d_levelgraph PRIVATE tensor)

  add_executable(bench_sem3d_levelgraph_li
    benchmarks/sem_stiffness_3d/bench_sem3d_levelgraph.cpp)
  target_link_libraries(bench_sem3d_levelgraph_li PRIVATE tensor)
  target_compile_options(bench_sem3d_levelgraph_li PRIVATE -lineinfo)
endif()
