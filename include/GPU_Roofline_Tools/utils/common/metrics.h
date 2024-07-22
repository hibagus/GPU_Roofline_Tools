#pragma once

class metrics
{
  uint32_t   n_iter;       // Number of iterations
  uint32_t   n_wg;         // Number of workgroup launched
  uint32_t   n_tr_per_wg;  // Number of thread per workgroup
  uint64_t   n_flops;      // Number of Flops
  uint64_t   n_bytes;      // Number of Bytes
  double     time_ms;      // Elapsed time in ms
};