#pragma once
#include <hip/hip_runtime.h>

template<typename TCompute>
__global__ void fma(TCompute *buf_A, TCompute *buf_B, TCompute *buf_C, uint64_t *dev_n_clockCount, uint32_t wf_sz)
{
    // Global Index (NDRange-level)
    const uint32_t thread_id     = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    const uint32_t wavefront_id  = thread_id / wf_sz;

    // Local Index (Workgroup-level)
    // This is to determine whether a thread is the 0th thread in a waveform (i.e., lane 0)
    const uint32_t lane_id_x = hipThreadIdx_x & (wf_sz - 1);

    uint64_t start_time;
    uint64_t end_time;

    // Load operand from Memory to Register
    TCompute a = buf_A[thread_id];
    TCompute b = buf_B[thread_id];
    TCompute c = buf_C[thread_id];

    // Compute
    start_time = clock64();
    #pragma unroll
    for(int iter=0; iter < NUM_LOOPS; iter++)
    {
        a = a * b + c;
    }
    end_time = clock64();

    // Store back the result to memory
    buf_A[thread_id] = a;

    if(lane_id_x==0){dev_n_clockCount[wavefront_id] = end_time-start_time;}
}

template<typename TCompute>
__global__ void fma(TCompute *buf_A, const TCompute B, TCompute *buf_C, uint64_t *dev_n_clockCount, uint32_t wf_sz)
{
    // Global Index (NDRange-level)
    const uint32_t thread_id     = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    const uint32_t wavefront_id  = thread_id / wf_sz;

    // Local Index (Workgroup-level)
    // This is to determine whether a thread is the 0th thread in a waveform (i.e., lane 0)
    const uint32_t lane_id_x = hipThreadIdx_x & (wf_sz - 1);

    uint64_t start_time;
    uint64_t end_time;

    // Load operand from Memory to Register
    TCompute a = buf_A[thread_id];
    TCompute c = buf_C[thread_id];

    // Compute
    start_time = clock64();
    #pragma unroll
    for(int iter=0; iter < NUM_LOOPS; iter++)
    {
        a = a * B + c;
    }
    end_time = clock64();

    // Store back the result to memory
    buf_A[thread_id] = a;

    if(lane_id_x==0){dev_n_clockCount[wavefront_id] = end_time-start_time;}
}

template<typename TCompute>
__global__ void fma(TCompute *buf_A, const TCompute B, uint64_t *dev_n_clockCount, uint32_t wf_sz)
{
    // Global Index (NDRange-level)
    const uint32_t thread_id     = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    const uint32_t wavefront_id  = thread_id / wf_sz;

    // Local Index (Workgroup-level)
    // This is to determine whether a thread is the 0th thread in a waveform (i.e., lane 0)
    const uint32_t lane_id_x = hipThreadIdx_x & (wf_sz - 1);

    uint64_t start_time;
    uint64_t end_time;

    // Load operand from Memory to Register
    TCompute a = buf_A[thread_id];

    start_time = clock64();
    #pragma unroll
    for(int iter=0; iter < NUM_LOOPS; iter++)
    {
        a = a * B + a;
    }
    end_time = clock64();

    // Store back the result to memory
    buf_A[thread_id] = a;

    if(lane_id_x==0){dev_n_clockCount[wavefront_id] = end_time-start_time;}
}

template<typename TCompute>
__global__ void fma(TCompute *buf_A, const TCompute B, const TCompute C, uint64_t *dev_n_clockCount, uint32_t wf_sz)
{
    // Global Index (NDRange-level)
    const uint32_t thread_id     = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    const uint32_t wavefront_id  = thread_id / wf_sz;

    // Local Index (Workgroup-level)
    // This is to determine whether a thread is the 0th thread in a waveform (i.e., lane 0)
    const uint32_t lane_id_x = hipThreadIdx_x & (wf_sz - 1);

    uint64_t start_time;
    uint64_t end_time;

    // Load operand from Memory to Register
    TCompute a = buf_A[thread_id];

    start_time = clock64();
    #pragma unroll
    for(int iter=0; iter < NUM_LOOPS; iter++)
    {
        a = a * B + C;
    }
    end_time = clock64();

    // Store back the result to memory
    buf_A[thread_id] = a;

    if(lane_id_x==0){dev_n_clockCount[wavefront_id] = end_time-start_time;}
}