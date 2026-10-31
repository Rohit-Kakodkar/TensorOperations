# Repository guidance

These instructions apply to all agents and sessions working in this repository.
Check for more specific `AGENTS.md` or `AGENTS.override.md` files in directories
you work in. Keep role-specific procedures in skills or agent definitions.

## Repository map

- `include/TensorOperations/`: header-only C++20 tensor library, exposed through
  the CMake interface target `tensor`. Graphs, labelled operands, layouts, and
  evaluators support Kokkos execution, with an optional CUDA/CuTe backend.
- `tests/`: GoogleTest runtime tests and compile-time assertions.
  `tests/negative/` exercises rejected configurations and runtime guards;
  registration and expected diagnostics live in `CMakeLists.txt`.
- `sfpp_min/`: a minimal SPECFEM++ stiffness problem, reference implementations,
  library-backed implementations, tests, and benchmarks.
- `benchmarks/`: tensor-library performance experiments.
- `CMakeLists.txt`: build targets, backend conditions, dependencies, and tests.
  `pyproject.toml` and `uv.lock` provide development tools, not the tensor runtime.
- `FUSED_OPERAND_STAGING_ANALYSIS.md` and `SEM_STIFFNESS_GAP_ANALYSIS.md` contain
  historical investigations. Check their claims against current code and tests.
- Build directories, fetched dependencies, and `.venv/` are generated artifacts;
  make source changes in the tracked project files.

## Working conventions

- Inspect `git status` and existing diffs before editing. Preserve unrelated
  work and keep changes focused on the current task.
- Read the relevant implementation and tests before changing an API or explaining
  its behavior. Surface disagreements between code, tests, and documentation.
- Preserve label/axis order, extent and stride semantics, memory-space access,
  synchronization, and scratch-storage lifetime/alignment contracts. When changing
  these, inspect the corresponding evaluator and supported backend paths.
- Keep compile-time guards and their diagnostics specific. Update the matching
  negative tests when intentionally changing a guard or its wording.
- Preserve the independence of `sfpp_min` reference/baseline implementations from
  the tensor implementations they validate or benchmark.

## C++ conventions

Use C++20 and the repository conventions below for new or modified code. Existing
APIs contain naming variations; preserve established public names and avoid
unrelated style migrations. `.clang-format` defines formatting, not the entire
Google C++ style guide.

### Naming and source organization

| Element | Convention and repository examples |
| --- | --- |
| Public types and concepts | `PascalCase`: `TensorHandle`, `SlotStore`, `TensorLike` |
| Functions and factories | `snake_case`: `make_handle`, `scratch_bytes`, `tuple_append` |
| Variables and parameters | Descriptive `snake_case`; short mathematical indices in small loops |
| Type aliases and traits | Preserve family conventions: `value_type`, `exec_space`, `tuple_element_t`, `tuple_size_v` |
| Compile-time metadata | Match the existing interface: `Rank`, `NumOut`, `num_levels`, `is_static`; no blanket renaming |
| Files | `.hpp` for library headers, `.cpp` for tests/benchmarks; `test_<feature>.cpp` for tests |
| Macros | Uppercase project prefixes such as `TENSOR_OPS_` and `SFPP_MIN_` |

- Use `#pragma once` in headers. Include the declarations you use directly;
  do not depend on accidental transitive includes. Use
  `<TensorOperations/Header.hpp>` for library includes and preserve local include
  grouping; automatic include sorting is disabled.
- Put library APIs in `TensorOperations` and private machinery in
  `TensorOperations::Impl`. Keep test-only helpers in unnamed namespaces in `.cpp`
  files. Do not add namespace-wide `using namespace` directives to public headers.
- Refer to the library from outside it through the alias
  `namespace TOps = TensorOperations;`, and write `TOps::make_handle`. Put the
  alias at file scope in `.cpp` files, benchmarks, `sfpp_min` sources, and
  documentation examples. In a header, declare it inside the function that
  uses it, or write `TensorOperations::` in full; a header-level alias or
  `using namespace` leaks into every includer. Tests and negative tests under
  `tests/` may keep `using namespace TensorOperations;` to keep assertions short.
- Keep header definitions safe across translation units: templates, class-defined
  functions, `constexpr` functions, or explicitly `inline` free functions as
  appropriate. Use `inline constexpr` for shared header constants. Avoid mutable
  global state and non-inline external definitions in headers.

### Interfaces, ownership, and types

- Prefer value types and the rule of zero. Use RAII for host-owned resources and
  Kokkos views for managed tensor storage. Treat raw pointers and unmanaged views
  as borrowed storage; document who owns it and how long it must remain valid.
- Pass small scalars and lightweight handles by value when appropriate; use
  `const T&` for larger read-only inputs and `T&` for deliberate mutation. Follow
  Kokkos value-capture patterns for kernels rather than capturing host references.
- Make non-mutating member functions `const`. Remember that a const Kokkos view
  handle does not make its elements const; use a const element type when the API
  promises read-only data. Never return references to local or expired storage.
- Make single-argument converting constructors `explicit` unless conversion is
  intentional. Add `noexcept` only when the operations actually guarantee it;
  use `[[nodiscard]]` where discarding a result would lose a required outcome.
- Prefer `using` aliases, `nullptr`, and named C++ casts. Use `auto` for iterator,
  expression-template, and factory results; spell types when numeric precision,
  ownership, or signedness is important to the contract.
- Preserve established index types: labels use `int32_t`, many ranks/extents use
  `int`, and byte counts use `std::size_t`. Check bounds before narrowing; choose
  arithmetic types that avoid overflow and signed/unsigned comparison errors.

### Templates and contracts

- Reuse existing concepts and traits. Prefer `requires` for overload eligibility,
  `if constexpr` for type-dependent branches, and `static_assert` for invalid
  compile-time configurations with actionable diagnostics.
- Use `constexpr` for operations usable during constant evaluation and `consteval`
  only when evaluation must be compile-time. Keep static shape/label information
  in types or constants instead of adding runtime dispatch to established paths.
- Preserve documented compiler workarounds in `Concept.hpp` and related headers.
  Simplifying SFINAE or pack expansions requires validation on the affected
  compilers, including nvcc when applicable.
- Use assertions for internal invariants, accounting for their removal under
  `NDEBUG`. Required runtime input checks must remain effective in release builds;
  follow the existing host/device error path rather than adding device exceptions.

### Kokkos and numerical code

- Mark device-executed functions with the appropriate `KOKKOS_FUNCTION`,
  `KOKKOS_INLINE_FUNCTION`, or existing project macro. Use
  `KOKKOS_DEFAULTED_FUNCTION` where needed for device-callable defaulted members.
  Compile-time host-only planners, such as the reshape planner in `TileLayout.hpp`,
  intentionally remain unannotated.
- Use device-compatible containers and callables. Follow `DeviceTuple.hpp` for
  heterogeneous device storage; do not replace it with runtime `std::tuple`
  operations without proving backend compatibility. Keep host-only allocation,
  I/O, and exceptions out of device paths.
- Respect execution/memory spaces, team barriers, and scratch lifetimes. Add
  fences at required synchronization or measurement boundaries, with a reason;
  do not insert them indiscriminately into hot paths.
- Preserve scalar precision and numerical semantics. Changes to reduction order,
  contraction order, or accumulation types require reference comparisons and
  justified tolerances. Do not relax tolerances simply to make a test pass.
- Reuse existing layouts, evaluators, and storage utilities. Introduce an
  abstraction when it captures a real shared contract, and keep allocation and
  unnecessary copies out of inner evaluation loops.

## Build and validation

Use an out-of-source build directory appropriate to the compiler and backend.
The project requires CMake 3.22 or newer and a C++20 compiler. Configuration uses
FetchContent for Kokkos (`develop`) and GoogleTest; it may need network access.
Reuse existing dependency checkouts when available rather than editing them.

Example CPU/Serial smoke check from the repository root:

```sh
cmake -S . -B build_agent -DCMAKE_BUILD_TYPE=Debug -DKokkos_ENABLE_SERIAL=ON
cmake --build build_agent --target test_concept test_level_graph --parallel 2
ctest --test-dir build_agent -R '^(ConceptTest|LevelPlanTest|LevelGraph)' --output-on-failure --no-tests=error
```

- The smoke check covers only those targets. Build the targets relevant to your
  change and run their registered tests; use `ctest --test-dir build_agent -N`
  to inspect test names. Broaden validation for changes to shared infrastructure.
- Prefer explicit build targets on CPU: some profiling targets use CUDA-only
  `-lineinfo` flags even with `TENSOR_OPS_BUILD_BENCHMARKS=OFF`.
- Run negative compilation tests through CTest, which checks their intended
  diagnostic. For example:
  `ctest --test-dir build_agent -R '^LabelTilesGuardNegative\.' --parallel 1 --output-on-failure --no-tests=error`.
  These tests invoke builds themselves; run them serially in a shared build tree.
- CUDA/CuTe requires a suitable CUDA/Kokkos toolchain. CuTe targets are enabled
  when both `Kokkos_ENABLE_CUDA` and `TENSOR_OPS_CUTLASS_DIR` are set. Report which
  backends were actually validated; CPU success does not establish CUDA success.
- For behavior changes, add or update tests that exercise the observable contract.
  Use distinct axis extents and non-self-inverse permutations when testing layout
  transformations so symmetric cases cannot conceal ordering bugs.
- Follow nearby Kokkos test initialization/finalization and CMake registration;
  many runtime tests use a custom `main` and link `GTest::gtest`.
- Prose-only changes need factual, reference, and diff checks. Compile or run new
  executable examples when practical; report verification limits explicitly.

## Formatting

Use `.clang-format`: Google-based formatting, aligned declarations/assignments,
and include sorting disabled. Format changed C++ files without reformatting
unrelated files. Tooling requires Python 3.12 or newer:

```sh
uv sync --locked
uv run clang-format --dry-run --Werror include/TensorOperations/TensorHandle.hpp
```

Replace the example path with the changed C++ files; use `-i` to apply formatting.
The repository's pre-commit hook also runs clang-format. Use the locked tool
version rather than introducing a separate formatter configuration.

## Documentation is for people

Documentation here is read by people with a goal, not parsed by tools. Every
sentence should help someone do something or understand something; if it does
neither, cut it.

- Start from the reader's question or task, not from the API's structure. Say
  what to do and when, then why. When two options exist, give the decision rule.
- Be direct and concrete: plain active voice, "you" for the reader, realistic
  example values, and one short compiled example in place of paragraphs of
  description. Name the common mistake and how it shows up.
- Define a term where it first appears and use one word for one idea.
- Be proportionate. Do not restate a signature, a name, or a type. Do not fill in
  a tag or section because the standard lists it; omit what carries no
  information. A short, accurate comment beats a complete, padded one.
- Match the voice of the neighbouring documentation. Read it before adding to it.
- Clarity to a reader outranks tag completeness and uniform structure. The
  Doxygen standard below is a format convention for API comments; where it would
  force padding, drop the padding.
- Documentation states what is true and what to do. It does not narrate how it
  was written or checked: no "we tried", "the trial showed", "was not run", and
  no replies to reviewer comments. How it was verified belongs in the handoff,
  PR description, or task file. A limit stays only when it changes what the
  reader should do, and then it is written as a fact about the software or
  platform ("verified on macOS with Serial; other platforms are untried"), once.
- Gaps are measured by where a person would get stuck, guess wrong, or have to
  read source or tests, not by which symbols lack comments.

## Documentation and evidence

- Write for the reader's task. Explain contracts, assumptions, and reasoning
  that require investigation; avoid comments that merely repeat the code.
- Put API contracts near declarations and implementation reasoning near the code.
  Use broader guides or examples for explanations spanning several components.
- Verify examples, symbols, and links. State prerequisites and backend limits
  where they affect usage. Follow the Doxygen standard below for API comments.
- Support behavior claims with source and tests. Distinguish known behavior from
  inferred intent, and identify uncertainty rather than inventing rationale.
- Support performance claims with measurements and their hardware, backend,
  compiler/build configuration, problem sizes, and correctness checks. Keep
  historical measurements tied to their recorded configuration.

### Doxygen standard

Use Doxygen-compatible comments for new public APIs and when substantively changing
an existing public contract. Cover public types, concepts, factories, functions,
and members whose meaning is not obvious. Update documentation with the contract;
do not rewrite unrelated comments just to change their syntax.

**Placement and format**

- Put a `/** ... */` block immediately before the declaration, above its
  `template` line. Use a leading ` * ` on each content line and `@` commands
  consistently. Write complete sentences and separate paragraphs with a blank
  ` *` line. Wrap prose near 80 columns; keep identifiers and URLs intact.
- Start every full API block with an explicit, one-sentence `@brief`. Follow with
  explanatory paragraphs when needed. Describe semantics rather than restating
  the signature. Use backticks for identifiers and `@ref` for resolvable links.
- Use `/// @brief ...` for a genuinely short declaration comment and `///< ...`
  for a short data-member or enumerator description. Use ordinary `//` comments
  inside function bodies for implementation reasoning.
- Document the canonical declaration once. For overloads, explain differences;
  use `@copydoc` only when the entire contract and parameter names really match.
  Avoid detached `@fn`/`@class` declarations that duplicate C++ signatures.

**Structure and required content**

Order full API blocks as follows, omitting sections that do not apply:

1. `@brief`, then the detailed behavior and mathematical meaning.
2. `@tparam` entries in template declaration order, including parameter packs.
   Explain each parameter's role and constraints, not just that it is a type.
3. `@param[in]`, `@param[out]`, or `@param[in,out]` entries in function parameter
   order. Names must exactly match the declaration. Describe shape, label order,
   units, mutation, and aliasing where relevant. A by-value view can still be an
   output parameter when the function writes through it.
4. `@return` for non-void functions: explain the result's meaning, ownership,
   lifetime, and relationship to inputs. Use `@retval` for specific status values.
   Omit return tags for constructors and void functions.
5. `@pre` and `@post` for caller obligations and observable guarantees.
   Use `@par Constraints` for compile-time requirements and explain how invalid
   configurations are rejected. Distinguish these from runtime checks and from
   assertions that disappear in release builds.
6. `@note` for execution-space requirements, synchronization, storage lifetime,
   or complexity; `@warning` for concrete misuse hazards. Use `@throws` only for
   exceptions the implementation actually throws.
7. `@par Example` with `@code{.cpp}` / `@endcode` when usage is nontrivial, followed
   by `@see` for related APIs. Include necessary setup or explicitly state the
   example's prerequisites; verify code against the current API.

File documentation belongs after `#pragma once` and before includes: use `@file`
and `@brief` to explain the header's responsibility and entry points, following
`DeviceTuple.hpp`. Type documentation should explain representation, ownership,
and invariants (`@invariant` where appropriate). Concept documentation should
explain what satisfying the concept permits. Put multi-component tutorials in
Markdown guides, with links to the API; define groups centrally before using
`@ingroup`. Avoid adding author/date boilerplate that duplicates Git history.

For example, documentation for the existing `make_handle` contract could be:

```cpp
/**
 * @brief Associates compile-time mode labels with a tensor.
 *
 * Labels follow the tensor's axis order; labelling does not permute its data.
 *
 * @tparam Modes Distinct labels, one per tensor axis, in axis order.
 * @tparam T Tensor type satisfying TensorLike.
 * @param[in] base Tensor to wrap, passed by value.
 * @return A labelled handle containing the supplied tensor object. Storage
 * ownership follows T's copy/move semantics; borrowed storage remains borrowed.
 * @par Constraints
 * The number of labels must equal T::rank and all labels must be distinct.
 * TensorHandle rejects violations with static assertions.
 */
template <int32_t... Modes, TensorLike T>
auto make_handle(T base);
```

**Review and validation**

Check every tag against the signature and every behavioral claim against source
and tests. Check reference targets, balanced code blocks, and example prerequisites.
When a Doxygen build is available, inspect rendered output and resolve warnings
introduced by the change, especially unmatched parameters and unresolved links.
This repository currently has no Doxygen build configuration; report source-only
validation accurately and keep generator setup as a separately scoped change.

Syntax references: [comment blocks](https://www.doxygen.nl/manual/docblocks.html)
and [commands](https://www.doxygen.nl/manual/commands.html). The ordering and
coverage rules above are this repository's conventions.

## Review and handoff

- Review the final diff for scope, correctness, and accidental behavior changes.
  Separate actionable correctness findings from optional style suggestions.
- Summarize what changed and why, the checks run and their results, and remaining
  uncertainties or unavailable environments. Do not report unrun checks as passed.
- Leave changes local and uncommitted unless the current task requests a commit
  or publication. Keep these shared instructions accurate when project commands
  or conventions change.
