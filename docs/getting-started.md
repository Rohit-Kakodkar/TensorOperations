# Getting started

TensorOperations is header-only and is consumed through the CMake interface
target `tensor`. It requires CMake 3.22 or newer, a C++20 compiler and Kokkos.

## Build and run a smoke check

```sh
cmake -S . -B build_agent -DCMAKE_BUILD_TYPE=Debug -DKokkos_ENABLE_SERIAL=ON
cmake --build build_agent --target test_concept test_level_graph --parallel 2
ctest --test-dir build_agent -R '^(ConceptTest|LevelPlanTest|LevelGraph)' \
  --output-on-failure --no-tests=error
```

Configuration fetches Kokkos (`develop`) and GoogleTest with `FetchContent`, so
it may need network access.

## Use it in your own project

The commands above build this repository's tests. To link your own program to
`tensor`, see {doc}`howto/use-in-your-project`. It covers the routes that work
with or without a Kokkos of your own, keeping this repository's tests and
benchmarks out of your build, and running only your own tests.

## Backends

See {doc}`reference/backends` for what each backend requires and what has been
validated.

## Next steps

- Link your own program: {doc}`howto/use-in-your-project`.
- Write your first program: {doc}`tutorials/first-level-graph`.
- Solve a specific task with the {doc}`how-to guides <howto/index>`.
