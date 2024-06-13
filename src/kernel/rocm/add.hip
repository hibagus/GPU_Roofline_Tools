#include <hip/hip_runtime.h>
#include <GPU_Roofline_Tools/kernel/rocm/add.hpp>

//template<typename T>
__global__ void add_benchmark_memory(double *buf_A, double *buf_B, double *buf_C, uint32_t size, uint32_t n_loop)
{
    const uint32_t thread_id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if (thread_id < size)
    {
        for(uint32_t iter=0; iter < n_loop; iter++)
        {
            buf_C[thread_id] = buf_A[thread_id] + buf_B[thread_id];
        }
    }
}