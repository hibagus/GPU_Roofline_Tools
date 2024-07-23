#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

template<typename TCompute>
__global__ void add(TCompute *buf_A, TCompute *buf_B, uint32_t n_loop, uint64_t *dev_n_clockCount, uint32_t wf_sz)
{
    // Global Index (NDRange-level)
    const uint32_t thread_id     = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    const uint32_t wavefront_id  = thread_id / wf_sz;

    // Local Index (Workgroup-level)
    // This is to determine whether a thread is the 0th thread in a waveform (i.e., lane 0)
    const uint32_t lane_id_x = hipThreadIdx_x & (wf_sz - 1);

    uint64_t start_time;
    uint64_t end_time;

    start_time = clock64();

    #pragma unroll
    for(uint32_t iter=0; iter < n_loop; iter++)
    {
        buf_A[thread_id] = buf_A[thread_id] + buf_B[thread_id];
    }

    end_time = clock64();

    if(lane_id_x==0){dev_n_clockCount[wavefront_id] = end_time-start_time;}

}
