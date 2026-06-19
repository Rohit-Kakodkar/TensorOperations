#pragma once
#include <Kokkos_Core.hpp>
#include <array>
#include <cstdio>
#include <atomic>

namespace TensorOperations {

// Global timing counters for each evaluator operator stage.
// Thread-safe via atomics (though Kokkos may serialize on host).
struct TimingStats {
  // Input staging (RangePolicyTag + InputTag specialization)
  std::atomic<long long> input_stage_load_count{0};
  std::atomic<long long> input_stage_load_time{0};

  // Contraction (RangePolicyTag + ContractionTag specialization)
  std::atomic<long long> contraction_accum_count{0};
  std::atomic<long long> contraction_accum_time{0};
  std::atomic<long long> contraction_block_load_count{0};
  std::atomic<long long> contraction_block_load_time{0};
  std::atomic<long long> contraction_compute_count{0};
  std::atomic<long long> contraction_compute_time{0};

  // Store (RangePolicyTag + IntermTag + RegisterArray specialization)
  std::atomic<long long> store_write_count{0};
  std::atomic<long long> store_write_time{0};

  // Scratch tier (TeamPolicyTag) — input load
  std::atomic<long long> scratch_input_load_count{0};
  std::atomic<long long> scratch_input_load_time{0};

  // Scratch tier (TeamPolicyTag) — contraction
  std::atomic<long long> scratch_contraction_count{0};
  std::atomic<long long> scratch_contraction_time{0};

  // Reset all counters
  void reset() {
    input_stage_load_count       = 0;
    input_stage_load_time        = 0;
    contraction_accum_count      = 0;
    contraction_accum_time       = 0;
    contraction_block_load_count = 0;
    contraction_block_load_time  = 0;
    contraction_compute_count    = 0;
    contraction_compute_time     = 0;
    store_write_count            = 0;
    store_write_time             = 0;
    scratch_input_load_count     = 0;
    scratch_input_load_time      = 0;
    scratch_contraction_count    = 0;
    scratch_contraction_time     = 0;
  }

  // Print a human-readable report to stdout
  void print_report() const {
    std::printf("\n=== TensorOperations Timing Report ===\n");
    std::printf("%-40s %15s %15s\n", "Operation", "Count", "Time (ns)");
    std::printf("%-40s %15s %15s\n", "----------------------------------------",
                "---------------", "---------------");

    if (input_stage_load_count > 0) {
      std::printf("%-40s %15lld %15lld\n", "Input Stage Load",
                  input_stage_load_count.load(), input_stage_load_time.load());
    }

    if (contraction_block_load_count > 0) {
      std::printf("%-40s %15lld %15lld\n", "Contraction Block Load (A+B)",
                  contraction_block_load_count.load(),
                  contraction_block_load_time.load());
    }

    if (contraction_compute_count > 0) {
      std::printf("%-40s %15lld %15lld\n", "Contraction Compute (GEMM)",
                  contraction_compute_count.load(),
                  contraction_compute_time.load());
    }

    if (contraction_accum_count > 0) {
      std::printf("%-40s %15lld %15lld\n", "Contraction Accumulation (total)",
                  contraction_accum_count.load(),
                  contraction_accum_time.load());
    }

    if (store_write_count > 0) {
      std::printf("%-40s %15lld %15lld\n", "Store Write",
                  store_write_count.load(), store_write_time.load());
    }

    if (scratch_input_load_count > 0) {
      std::printf("%-40s %15lld %15lld\n", "Scratch Input Load",
                  scratch_input_load_count.load(),
                  scratch_input_load_time.load());
    }

    if (scratch_contraction_count > 0) {
      std::printf("%-40s %15lld %15lld\n", "Scratch Contraction",
                  scratch_contraction_count.load(),
                  scratch_contraction_time.load());
    }

    std::printf("\n");
  }
};

// Global timing statistics instance
inline TimingStats g_timing_stats;

// RAII timer for measuring wall-clock time in nanoseconds
class ScopedTimer {
 public:
  explicit ScopedTimer(std::atomic<long long>& accumulator)
      : accumulator_(accumulator) {
    start_ = Kokkos::Impl::clock_tic();
  }

  ~ScopedTimer() {
    uint64_t end = Kokkos::Impl::clock_tic();
    // Accumulates raw CPU CYCLES (clock_tic), not nanoseconds. Consumers must
    // treat these counters as fractions and rescale to a real wall-clock anchor
    // (see TimingAnalyzer::print_detailed_analysis) — they are not directly ms.
    accumulator_ += static_cast<long long>(end - start_);
  }

 private:
  uint64_t                start_;
  std::atomic<long long>& accumulator_;
};

}  // namespace TensorOperations

// Timing instrumentation macros (disabled by default; enable with
// TENSOR_OPS_ENABLE_TIMING)
#ifdef TENSOR_OPS_ENABLE_TIMING
#define TIMING_SCOPE_ENTER(counter_time, counter_count) \
  uint64_t _timer_start_ = Kokkos::Impl::clock_tic();

#define TIMING_SCOPE_EXIT(counter_time, counter_count)       \
  do {                                                       \
    uint64_t _timer_end_ = Kokkos::Impl::clock_tic();        \
    counter_time.fetch_add(                                  \
        static_cast<long long>(_timer_end_ - _timer_start_), \
        std::memory_order_relaxed);                          \
    counter_count.fetch_add(1, std::memory_order_relaxed);   \
  } while (false);
#else
#define TIMING_SCOPE_ENTER(counter_time, counter_count) (void)0
#define TIMING_SCOPE_EXIT(counter_time, counter_count) (void)0
#endif
