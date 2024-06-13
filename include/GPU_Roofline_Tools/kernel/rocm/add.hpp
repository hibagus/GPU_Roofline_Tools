#pragma once
//template<typename T>
__global__ void add_benchmark_memory(double *buf_A, double *buf_B, double *buf_C, uint32_t size, uint32_t n_loop);