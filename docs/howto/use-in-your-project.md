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

Every route was tried with the indirect-access program, copied unchanged from
that page, and each printed `ok`. The trials ran on one machine: macOS arm64,
Homebrew clang 22.1.2, CMake 4.0.3, Release builds, Serial backend only.
Route 3 was also run with the install and the project on a case-sensitive APFS
volume. Other compilers, platforms and backends were not tried.

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
`-DKokkos_ENABLE_SERIAL=ON` (Serial was already the default in the trial).
`EXCLUDE_FROM_ALL` matters; see [Keep the tests out of your build](#keep-the-tests-and-benchmarks-out-of-your-build).

Trials: `add_subdirectory` from a local checkout, and `FetchContent` from a
`file://` clone pinned to a commit. Both configured, built `my_app` and ran.
`FetchContent` with `GIT_TAG main` was not run; it is the same mechanism.

## Route 2: you build Kokkos in your project

If your project already has a Kokkos, the library's own fetch collides with it.
What works depends on how you bring yours in.

- **Declare it with `FetchContent` under the name `kokkos`, before adding
  TensorOperations.** The first declaration of a name wins, so the library's
  `FetchContent_MakeAvailable(kokkos)` reuses yours. Tried, with a Kokkos from a
  source tree and with Kokkos `5.1.1` from its git tag.

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

The build used the installed Kokkos and fetched none. GoogleTest is still
fetched, because no installed copy was on the prefix.

Do not rely on `-DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=ALWAYS` instead. It asks
for `find_package(kokkos)` in lowercase, the name the library declares, while
Kokkos installs only `KokkosConfig.cmake`. On a case-insensitive file system
(the macOS default) that finds the install. On a case-sensitive one, with the
install on that file system, it does not, and CMake quietly fetches and builds
its own Kokkos `develop` into `_deps/kokkos-src`. Configuration still succeeds,
so check `kokkos_DIR` in `CMakeCache.txt`: it should be your install's
`lib/cmake/Kokkos`, not `.../CMakeFiles/pkgRedirects`.

Your own `find_package(Kokkos REQUIRED)` before the declaration is fine. Keep
it if you need it elsewhere; with the declaration after it, `kokkos_DIR` and
`Kokkos_DIR` both pointed at the install and only GoogleTest was fetched. What
fails is `find_package(Kokkos)` with no `FIND_PACKAGE_ARGS` declaration. Then the
library builds a second Kokkos in your project, and configuration stops with
`Given deprecated option list Kokkos_DEVICES`. Both cases were run with the
install and project on a case-sensitive APFS volume.

## Route 4: no `tensor` target at all

The library is header-only, so you can point at its headers and link Kokkos
yourself:

```cmake
find_package(Kokkos REQUIRED)
add_executable(my_app main.cpp)
target_include_directories(my_app PRIVATE path/to/TensorOperations/include)
target_link_libraries(my_app PRIVATE Kokkos::kokkos)
```

Tried with the indirect-access program against an installed Kokkos. You give up
what the `tensor` target adds (see `CMakeLists.txt`): the `cxx_std_20`
requirement, so set C++20 yourself if Kokkos does not; `-fbracket-depth=2048` on
Clang and AppleClang; and the `TENSOR_OPS_ENABLE_TIMING`,
`TENSOR_OPS_VERIFY_RESHAPE` and `TENSOR_OPS_ENABLE_CUTE` definitions. The
indirect-access program compiled without the bracket-depth flag. Whether larger
programs need it was not established; if Clang stops with a bracket nesting
error, add `-fbracket-depth=2048`.

## Keep the tests and benchmarks out of your build

The library's `CMakeLists.txt` defines about 40 test, `sfpp_min` and benchmark
executables whether or not it is the top-level project. It has no switch to turn
them off. Without `EXCLUDE_FROM_ALL`, they all join your `all` target; the
trial's `all` held 42 library executables, plus the Kokkos and GoogleTest
targets.

Add `EXCLUDE_FROM_ALL` as in the examples above. With it, `all` held only `my_app`
and the Kokkos targets it needs. The library's tests do not appear in your
`ctest` list either way unless you call `enable_testing()` yourself; the trial
did not, and `ctest -N` showed zero tests with and without the keyword.

Three of those targets (`sfpp_min_bench_li`, `sfpp_min_bench_new_li` and
`sfpp_min_bench_te_li`) pass `-lineinfo`, a CUDA compiler flag, even on a CPU
build. The Homebrew clang in the trial accepted it with a warning
(`-lineinfo: 'linker' input unused`), and `sfpp_min_bench_li` built and linked;
the other two were not built. Other compilers were not tried. `AGENTS.md` warns
contributors to build explicit targets on CPU because of these flags, so treat
a failure there as possible.

## Which Kokkos you get

Route 1 uses whatever Kokkos `develop` is when you configure: the library's
`CMakeLists.txt` pins `GIT_TAG develop`, not a tagged release. Routes 2 to 4 use
yours (Route 3 only with the `FIND_PACKAGE_ARGS` declaration). This page does
not give a minimum Kokkos version. The indirect-access program built and ran
against Kokkos `5.1.1` (Route 2); no other program or test was run against it.

## Pick a backend

Backends come from Kokkos, so set the Kokkos options on your configure line or
in the cache before the library is added, for example
`-DKokkos_ENABLE_OPENMP=ON`. Only Serial was tried; see
{doc}`../reference/backends` for what each backend needs and what has been
validated.

The CuTe backend needs `-DTENSOR_OPS_CUTLASS_DIR=/path/to/cutlass` and a Kokkos
CUDA build. In `CMakeLists.txt` the CUTLASS include path and the
`TENSOR_OPS_ENABLE_CUTE` definition are added to `tensor` only when both
`TENSOR_OPS_CUTLASS_DIR` and `Kokkos_ENABLE_CUDA` are set in the project that
adds the library. That is read from the source and was not run. With a
Kokkos you did not build in the same project (Routes 3 and 4), do not assume the
second condition holds.

## See also

- {doc}`../reference/build-options` for the library's CMake variables.
- {doc}`indirect-access` for the program used in these trials.
