#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <rocwmma/rocwmma.hpp>
#include <GPU_Roofline_Tools/utils/common/global.h>

template<typename TMul, typename TAcc, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void wmma(TMul *buf_A, TMul *buf_B, TAcc *buf_C, uint64_t *dev_n_clockCount, uint32_t wf_sz)
{
    // Global Index (NDRange-level)
    const uint32_t thread_id     = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    const uint32_t wavefront_id  = thread_id / wf_sz;

    // Local Index (Workgroup-level)
    // This is to determine whether a thread is the 0th thread in a waveform (i.e., lane 0)
    const uint32_t lane_id_x = hipThreadIdx_x & (wf_sz - 1);

    uint64_t start_time;
    uint64_t end_time;

    // WMMA is warp-level operation, so we are not really looking at thread-level activities
    rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, TMul, rocwmma::row_major> fragment_a;
    rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, TMul, rocwmma::col_major> fragment_b;
    rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, TAcc> fragment_c;

    // get pointer to matrix A and B; allocated to each wavefront
    TMul const* local_A = buf_A + wavefront_id*WMMA_M;
    TMul const* local_B = buf_B + wavefront_id*WMMA_K;
    TAcc *local_C = buf_C + wavefront_id*WMMA_N;
    
    // Load Fragment of Matrix A and B 
    rocwmma::load_matrix_sync(fragment_a, local_A, WMMA_K);
    rocwmma::load_matrix_sync(fragment_b, local_B, WMMA_K);

    // Initialize Accumulator and Output Matrix to 0
    rocwmma::fill_fragment(fragment_c, static_cast<TAcc>(0));

    // Compute
    start_time = clock64();
    #pragma unroll
    for(int iter=0; iter < NUM_LOOPS; iter++)
    {
        rocwmma::mma_sync(fragment_c, fragment_a, fragment_b, fragment_c);
    }
    end_time = clock64();

    // Save Result
    rocwmma::store_matrix_sync(local_C, fragment_c, WMMA_N, rocwmma::mem_row_major);
    //if(hipBlockIdx_x == 0)
    //   rocwmma::store_matrix_sync(local_C, fragment_c, WMMA_N, rocwmma::mem_row_major);

    if(lane_id_x==0){dev_n_clockCount[wavefront_id] = end_time-start_time;}

}