#pragma once

class metrics
{
  public:
  uint32_t   n_iter;       // Number of iterations
  uint32_t   n_thread;     // Number of threads
  uint32_t   n_wg;         // Number of workgroup launched
  uint32_t   n_wf;         // Number of wavefront launched
  uint32_t   wf_size;      // Number of thread per workgroup
  uint32_t   wg_size;      // Number of thread per workgroup
  uint64_t   n_flops;      // Number of Flops
  uint64_t   n_bytes;      // Number of Bytes
  double     time_ms;      // Elapsed time in ms
  double     avg_clock;    // Average clock cycle across all wavefronts
  uint64_t   max_clock;    // Max clock cycle across all wavefronts
  uint64_t   min_clock;    // Min clock cycle across all wavefronts
  double     stdev_clock;  // Standard deviation across all wavefronts
};