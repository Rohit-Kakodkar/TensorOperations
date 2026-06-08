cmake_minimum_required(VERSION 3.22)
project(TensorKokkos CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tensor INTERFACE)
target_include_directories(tensor INTERFACE include/)

enable_testing()

add_executable(test_concept tests/test_concept.cpp)
target_link_libraries(test_concept PRIVATE tensor)
add_test(NAME concept COMMAND test_concept)
