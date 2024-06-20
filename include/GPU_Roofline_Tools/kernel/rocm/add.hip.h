#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

template<typename TCompute>
__global__ void add(TCompute *buf_A, TCompute *buf_B, uint32_t size, uint32_t n_loop)
{
    const uint32_t thread_id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if (thread_id < size)
    {
        for(uint32_t iter=0; iter < n_loop; iter++)
        {
            buf_A[thread_id] = buf_A[thread_id] + buf_B[thread_id];
        }
    }
}
