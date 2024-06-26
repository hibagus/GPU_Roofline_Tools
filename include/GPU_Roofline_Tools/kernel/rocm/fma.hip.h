#pragma once
#include <hip/hip_runtime.h>

template<typename TCompute>
__global__ void fma(TCompute *buf_A, TCompute *buf_B, TCompute *buf_C, uint32_t size, uint32_t n_loop)
{
    const uint32_t thread_id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if (thread_id < size)
    {
        for(uint32_t iter=0; iter < n_loop; iter++)
        {
            buf_A[thread_id] = buf_A[thread_id] * buf_B[thread_id] + buf_C[thread_id];
        }
    }
}

template<typename TCompute>
__global__ void fma(TCompute *buf_A, const TCompute B, TCompute *buf_C, uint32_t size, uint32_t n_loop)
{
    const uint32_t thread_id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if (thread_id < size)
    {
        for(uint32_t iter=0; iter < n_loop; iter++)
        {
            buf_A[thread_id] = buf_A[thread_id] * B + buf_C[thread_id];
        }
    }
}

template<typename TCompute>
__global__ void fma(TCompute *buf_A, const TCompute B, uint32_t size, uint32_t n_loop)
{
    const uint32_t thread_id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if (thread_id < size)
    {
        for(uint32_t iter=0; iter < n_loop; iter++)
        {
            buf_A[thread_id] = buf_A[thread_id] * B + buf_A[thread_id];
        }
    }
}