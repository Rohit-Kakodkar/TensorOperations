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

# test_device_tuple — custom main() wraps Kokkos init/finalize
add_executable(test_device_tuple tests/test_device_tuple.cpp)
target_link_libraries(test_device_tuple PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_device_tuple)

add_executable(test_team tests/test_team.cpp)
target_link_libraries(test_team PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_team)

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

# test_slot_store — custom main() wraps Kokkos init/finalize
add_executable(test_slot_store tests/test_slot_store.cpp)
target_link_libraries(test_slot_store PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_slot_store)

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

# test_functional_input — a leaf whose elements are computed, not addressed.
# Custom main() wraps Kokkos init/finalize; every case launches a graph.
add_executable(test_functional_input tests/test_functional_input.cpp)
target_link_libraries(test_functional_input PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_functional_input)

# test_sink_combine — a combine that returns void, scatters to global itself,
# and leaves the graph with no outputs. Custom main() wraps Kokkos.
add_executable(test_sink_combine tests/test_sink_combine.cpp)
target_link_libraries(test_sink_combine PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_sink_combine)

set(TENSOR_OPS_CUTLASS_DIR "" CACHE PATH
    "CUTLASS checkout for the CuTe backend (CUDA builds only)")
if(TENSOR_OPS_CUTLASS_DIR AND Kokkos_ENABLE_CUDA)
  target_include_directories(tensor INTERFACE ${TENSOR_OPS_CUTLASS_DIR}/include)
  target_compile_definitions(tensor INTERFACE TENSOR_OPS_ENABLE_CUTE)

  add_executable(test_cute_input tests/test_cute_input.cpp)
  target_link_libraries(test_cute_input PRIVATE tensor GTest::gtest)
  gtest_discover_tests(test_cute_input)

  add_executable(test_cute_smem_load tests/test_cute_smem_load.cpp)
  target_link_libraries(test_cute_smem_load PRIVATE tensor GTest::gtest)
  gtest_discover_tests(test_cute_smem_load)

  add_executable(test_cute_store tests/test_cute_store.cpp)
  target_link_libraries(test_cute_store PRIVATE tensor GTest::gtest)
  gtest_discover_tests(test_cute_store)

  add_executable(test_cute_staged tests/test_cute_staged.cpp)
  target_link_libraries(test_cute_staged PRIVATE tensor GTest::gtest)
  gtest_discover_tests(test_cute_staged)

  add_executable(test_cute_contract tests/test_cute_contract.cpp)
  target_link_libraries(test_cute_contract PRIVATE tensor GTest::gtest)
  gtest_discover_tests(test_cute_contract)

  add_executable(test_cute_combine tests/test_cute_combine.cpp)
  target_link_libraries(test_cute_combine PRIVATE tensor GTest::gtest)
  gtest_discover_tests(test_cute_combine)

  add_executable(test_cute_level_graph tests/test_cute_level_graph.cpp)
  target_link_libraries(test_cute_level_graph PRIVATE tensor GTest::gtest)
  gtest_discover_tests(test_cute_level_graph)
endif()

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
# Negative-compilation tests for the FUNCTIONAL-INPUT guard. Same rationale and
# same caveats as the guards above.
#
# A functional input answers at a coordinate and has no address, so it is legal
# only as a stage operand. Without these guards the mistake still fails to
# compile -- but deep inside evaluator instantiation, in a diagnostic naming
# layout internals rather than the mistake. The contraction and combine cases
# are pinned separately because they are independent factories: one guard
# present and the other missing must not read as both working.
# ---------------------------------------------------------------------------
foreach(case 0 1 2)
  add_executable(functional_guard_neg_${case} EXCLUDE_FROM_ALL
                 tests/negative/functional_guard_negative.cpp)
  target_link_libraries(functional_guard_neg_${case} PRIVATE tensor)
  target_compile_definitions(functional_guard_neg_${case}
                             PRIVATE FUNCTIONAL_NEG_CASE=${case})
  add_test(NAME FunctionalGuardNegative.Case${case}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                   --target functional_guard_neg_${case} --config $<CONFIG>)
endforeach()
set_tests_properties(FunctionalGuardNegative.Case1
                     FunctionalGuardNegative.Case2
                     PROPERTIES PASS_REGULAR_EXPRESSION
                     "a functional input has no address")

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
foreach(case 0 1 2 3 4 5)
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
                     LevelGuardNegative.Case5
                     PROPERTIES PASS_REGULAR_EXPRESSION
                     "must share ONE iteration space")
set_tests_properties(LevelGuardNegative.Case4 PROPERTIES
                     PASS_REGULAR_EXPRESSION
                     "may not read a slot ITS OWN LEVEL")

# Benchmarks (opt-in: -DTENSOR_OPS_BUILD_BENCHMARKS=ON). No gtest; own main().
if(TENSOR_OPS_BUILD_BENCHMARKS)
  # --- 3D SEM stiffness on the LEVEL GRAPH (Sprint 5) -----------------------
  # Note what is ABSENT, and that it is the point: no GEMM tuning knobs. The
  # contraction is a plain dot product with no register blocking, so NGLL=5
  # needs no per-target retuning and TE=1 is legal -- neither was true of the
  # removed register-blocked GEMM, whose target had to pin MT/NT/NR and
  # therefore could not express TE=1 at all. TE=1 is where both this
  # benchmark's implementations are fastest.
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

# sfpp_min — a minimal, zero-dependency stand-in for SPECFEM++'s 3D elastic
# isotropic stiffness kernel and its data layouts. Links Kokkos ONLY, never
# `tensor`: the dummy kernel must stay uncontaminated by the library it will be
# compared against. See plans/sprint1/ and plans/minimal-sfpp-library-sprints.md.
add_library(sfpp_min INTERFACE)
target_include_directories(sfpp_min INTERFACE sfpp_min/)
target_link_libraries(sfpp_min INTERFACE Kokkos::kokkos)
target_compile_features(sfpp_min INTERFACE cxx_std_20)

add_executable(sfpp_min_gll sfpp_min/tests/test_gll.cpp)
target_link_libraries(sfpp_min_gll PRIVATE sfpp_min GTest::gtest)
gtest_discover_tests(sfpp_min_gll TEST_PREFIX sfpp_min.)

add_executable(sfpp_min_layout sfpp_min/tests/test_layout.cpp)
target_link_libraries(sfpp_min_layout PRIVATE sfpp_min GTest::gtest)
gtest_discover_tests(sfpp_min_layout TEST_PREFIX sfpp_min.)

add_executable(sfpp_min_mesh sfpp_min/tests/test_mesh.cpp)
target_link_libraries(sfpp_min_mesh PRIVATE sfpp_min GTest::gtest)
gtest_discover_tests(sfpp_min_mesh TEST_PREFIX sfpp_min.)

add_executable(sfpp_min_geometry sfpp_min/tests/test_geometry.cpp)
target_link_libraries(sfpp_min_geometry PRIVATE sfpp_min GTest::gtest)
gtest_discover_tests(sfpp_min_geometry TEST_PREFIX sfpp_min.)

add_executable(sfpp_min_reference sfpp_min/tests/test_reference.cpp)
target_link_libraries(sfpp_min_reference PRIVATE sfpp_min GTest::gtest)
gtest_discover_tests(sfpp_min_reference TEST_PREFIX sfpp_min.)

add_executable(sfpp_min_kernel sfpp_min/tests/test_kernel.cpp)
target_link_libraries(sfpp_min_kernel PRIVATE sfpp_min GTest::gtest)
gtest_discover_tests(sfpp_min_kernel TEST_PREFIX sfpp_min.)

add_executable(sfpp_min_bench sfpp_min/bench.cpp)
target_link_libraries(sfpp_min_bench PRIVATE sfpp_min)

add_executable(sfpp_min_bench_li sfpp_min/bench.cpp)
target_link_libraries(sfpp_min_bench_li PRIVATE sfpp_min)
target_compile_options(sfpp_min_bench_li PRIVATE -lineinfo)

# --- Sprint 5: the LEVEL GRAPH kernel over the sfpp_min data model ---------
# These are the first sfpp_min targets that link `tensor`: the dummy above must
# stay uncontaminated by the library it is compared against, but the new kernel
# IS that library, expressed over the same data structures.
add_executable(sfpp_min_kernel_new sfpp_min/tests/test_kernel_new.cpp)
target_link_libraries(sfpp_min_kernel_new PRIVATE sfpp_min tensor GTest::gtest)
gtest_discover_tests(sfpp_min_kernel_new TEST_PREFIX sfpp_min.)

add_executable(sfpp_min_bench_new sfpp_min/bench_new.cpp)
target_link_libraries(sfpp_min_bench_new PRIVATE sfpp_min tensor)

add_executable(sfpp_min_bench_new_li sfpp_min/bench_new.cpp)
target_link_libraries(sfpp_min_bench_new_li PRIVATE sfpp_min tensor)
target_compile_options(sfpp_min_bench_new_li PRIVATE -lineinfo)

# The TE (tile-width) sweep. Its own binary because TE is a compile-time
# parameter of the graph: folding it into bench_new would multiply that
# harness's whole axis matrix by the number of TE values.
add_executable(sfpp_min_bench_te sfpp_min/bench_te.cpp)
target_link_libraries(sfpp_min_bench_te PRIVATE sfpp_min tensor)

add_executable(sfpp_min_bench_te_li sfpp_min/bench_te.cpp)
target_link_libraries(sfpp_min_bench_te_li PRIVATE sfpp_min tensor)
target_compile_options(sfpp_min_bench_te_li PRIVATE -lineinfo)

add_executable(sfpp_min_bench_mat sfpp_min/bench_mat.cpp)
target_link_libraries(sfpp_min_bench_mat PRIVATE sfpp_min)
