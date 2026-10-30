# Use TensorOperations in your own CMake project

You have a program, like the one in {doc}`indirect-access`, and a CMake project
of your own. You need the `tensor` target in that project. There is no
`install()` rule and no `find_package(TensorOperations)`, so you bring the
repository into your build. Pick a route by whether you already have Kokkos.

| You have | Use |
| --- | --- |
| No Kokkos, and you want the library to fetch it | [Route 1](#route-1-let-the-library-fetch-kokkos) |
| Kokkos as a source tree or `FetchContent` dependency in your build | [Route 2](#route-2-you-build-kokkos-in-your-project) |
| An installed Kokkos (`find_package(Kokkos)` works) | [Route 3](#route-3-you-have-an-installed-kokkos) or [Route 4](#route-4-no-tensor-target-at-all) |

These routes are verified on macOS (arm64, clang 22, CMake 4.0) with the Serial
backend. Other compilers, platforms and backends are untried.

## Route 1: let the library fetch Kokkos

```cmake
cmake_minimum_required(VERSION 3.22)
project(my_app CXX)

add_subdirectory(path/to/TensorOperations tensor_ops EXCLUDE_FROM_ALL)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE tensor)
```

Use `FetchContent` instead of a local checkout if you prefer. This needs CMake
3.28 or newer for the `EXCLUDE_FROM_ALL` keyword:

```cmake
include(FetchContent)
FetchContent_Declare(tensor_ops
  GIT_REPOSITORY https://github.com/Rohit-Kakodkar/TensorOperations.git
  GIT_TAG        main
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(tensor_ops)
```

The library's `CMakeLists.txt` fetches Kokkos `develop` and GoogleTest, and
builds Kokkos with the options you set in the cache before the call. Choose a
backend with the usual Kokkos variables, for example
`-DKokkos_ENABLE_SERIAL=ON` (Serial is on by default).
`EXCLUDE_FROM_ALL` matters; see [Keep the tests and benchmarks out of your build](#keep-the-tests-and-benchmarks-out-of-your-build).

## Route 2: you build Kokkos in your project

If your project already has a Kokkos, the library's own fetch collides with it.
What works depends on how you bring yours in.

- **Declare it with `FetchContent` under the name `kokkos`, before adding
  TensorOperations.** The first declaration of a name wins, so the library's
  `FetchContent_MakeAvailable(kokkos)` reuses yours. This works with a Kokkos from
  a source tree or from a git tag.

  ```cmake
  include(FetchContent)
  FetchContent_Declare(kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        5.1.1)
  FetchContent_MakeAvailable(kokkos)

  add_subdirectory(path/to/TensorOperations tensor_ops EXCLUDE_FROM_ALL)
  ```

- **`add_subdirectory(kokkos)` directly: does not work.** The library adds
  Kokkos again and configuration stops with `another target with the same name
  already exists` for Kokkos targets.

## Route 3: you have an installed Kokkos

Declare Kokkos yourself with `FIND_PACKAGE_ARGS NAMES Kokkos` before adding the
library. This needs CMake 3.24 or newer. `FetchContent` then calls
`find_package(Kokkos)` first and downloads Kokkos only if that fails. The
library's later `FetchContent_Declare(kokkos ...)` is ignored because yours came
first.

The fallback is silent. If `CMAKE_PREFIX_PATH` is wrong or missing, CMake
fetches and builds Kokkos `develop` with no error. Check `kokkos_DIR` in
`CMakeCache.txt`, as described below.

```cmake
cmake_minimum_required(VERSION 3.24)
project(my_app CXX)

include(FetchContent)
FetchContent_Declare(kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG        develop
  FIND_PACKAGE_ARGS NAMES Kokkos)

add_subdirectory(path/to/TensorOperations tensor_ops EXCLUDE_FROM_ALL)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE tensor)
```

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/kokkos/install
```

CMake uses the installed Kokkos and fetches none. GoogleTest is still fetched.

Do not rely on `-DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=ALWAYS` instead. It asks
for `find_package(kokkos)` in lowercase, the name the library declares, while
Kokkos installs only `KokkosConfig.cmake`. On a case-insensitive file system
(the macOS default) that finds the install. On a case-sensitive one, with the
install on that file system, it does not, and CMake quietly fetches and builds
its own Kokkos `develop` into `_deps/kokkos-src`. Configuration still succeeds,
so check `kokkos_DIR` in `CMakeCache.txt`: it should be your install's
`lib/cmake/Kokkos`, not `.../CMakeFiles/pkgRedirects`.

Your own `find_package(Kokkos REQUIRED)` before the declaration is fine; keep it
if you need it elsewhere. What fails is calling `find_package(Kokkos)` without
the `FIND_PACKAGE_ARGS` declaration. The library then builds a second Kokkos in
your project, and configuration stops with
`Given deprecated option list Kokkos_DEVICES`.

## Route 4: no `tensor` target at all

The library is header-only, so you can point at its headers and link Kokkos
yourself:

```cmake
find_package(Kokkos REQUIRED)
add_executable(my_app main.cpp)
target_include_directories(my_app PRIVATE path/to/TensorOperations/include)
target_link_libraries(my_app PRIVATE Kokkos::kokkos)
```

You give up what the `tensor` target adds (see `CMakeLists.txt`): the `cxx_std_20`
requirement, so set C++20 yourself if Kokkos does not; `-fbracket-depth=2048` on
Clang and AppleClang; and the `TENSOR_OPS_ENABLE_TIMING`,
`TENSOR_OPS_VERIFY_RESHAPE` and `TENSOR_OPS_ENABLE_CUTE` definitions. The
indirect-access program compiles without the bracket-depth flag. If Clang stops
with a bracket nesting error in a larger program, add `-fbracket-depth=2048`.

## Keep the tests and benchmarks out of your build

The library's `CMakeLists.txt` defines about 40 test, `sfpp_min` and benchmark
executables whether or not it is the top-level project. It has no switch to turn
them off. Without `EXCLUDE_FROM_ALL`, they all join your `all` target.

Add `EXCLUDE_FROM_ALL` as in the examples above. With it, `all` holds only your
targets and the Kokkos targets they need.

`EXCLUDE_FROM_ALL` does not hide the library's tests from CTest. The library
calls `enable_testing()` itself, so if your project calls it too, about 80 of the
library's tests join your `ctest` list, whether or not you used
`EXCLUDE_FROM_ALL` and wherever you call `enable_testing()`. With
`EXCLUDE_FROM_ALL` their executables are not built, so plain `ctest` reports them as `*_NOT_BUILT (Not Run)` and exits
with an error. The 48 `*Negative.*` tests start builds of their own when run.
Select your own tests instead. Label them:

```cmake
enable_testing()
add_test(NAME my_app COMMAND my_app)
set_tests_properties(my_app PROPERTIES LABELS mine)
```

```sh
ctest --test-dir build -L mine
```

`-R '^my_app'` also works if your test names share a prefix. Neither selection
runs the library's tests, including the `Negative` ones.

Three of the library's benchmark targets (`sfpp_min_bench_li`, `sfpp_min_bench_new_li` and
`sfpp_min_bench_te_li`) pass `-lineinfo`, a CUDA compiler flag, even on a CPU
build. Clang 22 accepts it with a warning (`-lineinfo: 'linker' input unused`);
`EXCLUDE_FROM_ALL` keeps these targets out of your build.

## Which Kokkos you get

Route 1 uses whatever Kokkos `develop` is when you configure: the library's
`CMakeLists.txt` pins `GIT_TAG develop`, not a tagged release. Routes 2 to 4 use
yours (Route 3 only with the `FIND_PACKAGE_ARGS` declaration).

`CMakeLists.txt` explains the `develop` pin: the GEMM path needs a Kokkos SIMD
fix that is newer than the `5.1.1` tag. The `5.1.1` tag builds and runs the
indirect-access program. Nothing beyond that program is established for a tagged
release, so if you pin one, check the features you use.

## Pick a backend

Backends come from Kokkos, so set the Kokkos options on your configure line or
in the cache before the library is added, for example
`-DKokkos_ENABLE_OPENMP=ON`. See {doc}`../reference/backends` for what each
backend needs and what has been validated.

The CuTe backend needs `-DTENSOR_OPS_CUTLASS_DIR=/path/to/cutlass` and a Kokkos
CUDA build. In `CMakeLists.txt` the CUTLASS include path and the
`TENSOR_OPS_ENABLE_CUTE` definition are added to `tensor` only when both
`TENSOR_OPS_CUTLASS_DIR` and `Kokkos_ENABLE_CUDA` are set in the project that
adds the library. With a Kokkos you did not build in the same project
(Routes 3 and 4), do not assume the second condition holds.

## See also

- {doc}`../reference/build-options` for the library's CMake variables.
- {doc}`indirect-access` for a complete program to build.
