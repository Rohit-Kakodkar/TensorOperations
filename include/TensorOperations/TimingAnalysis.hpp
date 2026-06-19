#pragma once
#include <TensorOperations/TimingInstrumentation.hpp>
#include <cstdio>
#include <cmath>

namespace TensorOperations {

// Utility functions for analyzing timing data
class TimingAnalyzer {
 public:
  // Convert nanoseconds to milliseconds
  static double ns_to_ms(long long ns) { return static_cast<double>(ns) / 1e6; }

  // Convert nanoseconds to seconds
  static double ns_to_sec(long long ns) {
    return static_cast<double>(ns) / 1e9;
  }

  // Compute average time per operation in nanoseconds
  static double avg_ns_per_op(long long total_ns, long long count) {
    if (count == 0) return 0.0;
    return static_cast<double>(total_ns) / count;
  }

  // Print detailed timing analysis with percentages and breakdowns.
  //
  // `wall_ms` is the trustworthy total wall-clock time for the measured region
  // (from Kokkos::Timer in the benchmark). The per-stage counters below are RAW
  // CPU CYCLES (clock_tic), not nanoseconds, so we use them ONLY as fractions
  // and rescale each stage to a share of `wall_ms`. This makes the printed ms
  // correct regardless of CPU frequency / turbo. Two caveats this relies on:
  //   * With a multithreaded host backend the cycle counters are summed across
  //     threads; that is fine because we only ever use ratios of (also-summed)
  //     totals.
  //   * input_stage_load is recorded INSIDE contraction_block_load (the
  //     stage_a/stage_b calls open Spec-1's own timing scope), so it is a
  //     sub-component of block load and must NOT be added to the denominator
  //     again — doing so was the old double-count bug.
  static void print_detailed_analysis(const TimingStats& stats,
                                      double             wall_ms) {
    std::printf("\n=== Detailed Timing Analysis ===\n");

    const long long input_stage = stats.input_stage_load_time.load();
    const long long block_load  = stats.contraction_block_load_time.load();
    const long long compute     = stats.contraction_compute_time.load();
    const long long store       = stats.store_write_time.load();
    const long long scratch_in  = stats.scratch_input_load_time.load();
    const long long scratch_ct  = stats.scratch_contraction_time.load();

    // De-nested denominator: input_stage is excluded (nested in block_load).
    const long long denom =
        block_load + compute + store + scratch_in + scratch_ct;

    if (denom == 0 || wall_ms <= 0.0) {
      std::printf(
          "No timing data collected. Ensure TENSOR_OPS_ENABLE_TIMING is "
          "defined.\n");
      return;
    }

    std::printf("Total (wall) time: %.3f ms\n\n", wall_ms);
    std::printf("%-40s %15s %15s %12s\n", "Operation", "Time (ms)", "Avg (ns)",
                "Percent");
    std::printf("%-40s %15s %15s %12s\n",
                "----------------------------------------", "---------------",
                "---------------", "----------");

    // A top-level stage: its wall-time share is its cycle fraction of `denom`.
    auto stage_line = [&](const char* name, long long cyc, long long count) {
      if (count <= 0) return;
      const double ms     = wall_ms * static_cast<double>(cyc) / denom;
      const double pct    = 100.0 * static_cast<double>(cyc) / denom;
      const double avg_ns = (count > 0) ? ms * 1e6 / count : 0.0;
      std::printf("%-40s %15.3f %15.1f %11.1f%%\n", name, ms, avg_ns, pct);
    };

    if (stats.contraction_block_load_count > 0) {
      const long long c = stats.contraction_block_load_count.load();
      stage_line("Contraction Block Load (A+B)", block_load, c);

      // Sub-components of block load (percent is relative to block load, not
      // wall): the nested global gather and the remaining copy/overhead (the
      // .storage_ copy and evaluator construction).
      if (stats.input_stage_load_count > 0 && block_load > 0) {
        const double block_ms =
            wall_ms * static_cast<double>(block_load) / denom;
        const long long gather =
            input_stage < block_load ? input_stage : block_load;
        const double gather_ms =
            block_ms * static_cast<double>(gather) / block_load;
        std::printf("%-40s %15.3f %15s %11.1f%%\n",
                    "  \xe2\x94\x94 of which global gather", gather_ms, "",
                    100.0 * static_cast<double>(gather) / block_load);
        std::printf(
            "%-40s %15.3f %15s %11.1f%%\n",
            "  \xe2\x94\x94 staging copy/overhead", block_ms - gather_ms, "",
            100.0 * static_cast<double>(block_load - gather) / block_load);
      }
    }

    stage_line("Contraction Compute (GEMM)", compute,
               stats.contraction_compute_count.load());
    stage_line("Store Write", store, stats.store_write_count.load());
    stage_line("Scratch Input Load", scratch_in,
               stats.scratch_input_load_count.load());
    stage_line("Scratch Contraction", scratch_ct,
               stats.scratch_contraction_count.load());

    std::printf("\n");

    // Memory-vs-compute balance. The full load cost is block_load (which
    // already contains the gather) plus any scratch-tier input load.
    const long long compute_time = compute;
    const long long load_time    = block_load + scratch_in;

    if (compute_time > 0 && load_time > 0) {
      double ratio = static_cast<double>(compute_time) / load_time;
      std::printf("Compute-to-Load Ratio: %.2f\n", ratio);
      std::printf(
          "  (Ratio > 1.0 suggests compute-bound; < 1.0 suggests "
          "memory-bound)\n");
      if (ratio > 2.0) {
        std::printf(
            "  -> COMPUTE-BOUND: optimization should focus on arithmetic "
            "efficiency\n");
      } else if (ratio > 1.0) {
        std::printf("  -> BALANCED: both memory and compute are significant\n");
      } else {
        std::printf(
            "  -> MEMORY-BOUND: optimization should focus on reducing memory "
            "traffic\n");
      }
    }
  }
};

}  // namespace TensorOperations
