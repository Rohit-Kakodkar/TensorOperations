cmake_minimum_required(VERSION 3.22)
project(TensorKokkos CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        5.1.1
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

enable_testing()
include(GoogleTest)

# test_concept — no Kokkos runtime; GTest provides main()
add_executable(test_concept tests/test_concept.cpp)
target_link_libraries(test_concept PRIVATE tensor GTest::gtest_main)
gtest_discover_tests(test_concept)

# test_layout — no Kokkos runtime; GTest provides main()
add_executable(test_layout tests/test_layout.cpp)
target_link_libraries(test_layout PRIVATE tensor GTest::gtest_main)
gtest_discover_tests(test_layout)

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

# test_register_array — custom main() wraps Kokkos init/finalize
add_executable(test_register_array tests/test_register_array.cpp)
target_link_libraries(test_register_array PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_register_array)

# test_tiling — custom main() wraps Kokkos init/finalize
add_executable(test_tiling tests/test_tiling.cpp)
target_link_libraries(test_tiling PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_tiling)

# test_contraction_eval — custom main() wraps Kokkos init/finalize
add_executable(test_contraction_eval tests/test_contraction_eval.cpp)
target_link_libraries(test_contraction_eval PRIVATE tensor GTest::gtest)
gtest_discover_tests(test_contraction_eval)

# Benchmark: naive RangePolicy vs library tiled contraction
add_executable(bench_contraction benchmarks/bench_contraction.cpp)
target_link_libraries(bench_contraction PRIVATE tensor)