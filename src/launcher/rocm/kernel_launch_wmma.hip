#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <GPU_Roofline_Tools/kernel/rocm/wmma.hip.h>
#include <GPU_Roofline_Tools/launcher/rocm/kernel_launch_wmma.hip.h>
#include <GPU_Roofline_Tools/utils/rocm/hipcheck.hip.h>
#include <GPU_Roofline_Tools/utils/common/optype.h>
#include <GPU_Roofline_Tools/utils/common/metrics.h>
#include <GPU_Roofline_Tools/utils/common/global.h>


metrics kernel_launch_wmma_f32_16x16x32_fp8_fp8(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz)
{
    //FP8 compute, FP32 accumulate
   return kernel_launch_wmma<rocwmma::float8_fnuz_t,rocwmma::float32_t,16,16,32>(n_wavefront, n_workgroup, dev_wf_sz);
}

metrics kernel_launch_wmma_f32_16x16x32_bf8_bf8(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz)
{
    //BF8 compute, FP32 accumulate
    return kernel_launch_wmma<rocwmma::bfloat8_fnuz_t,rocwmma::float32_t,16,16,32>(n_wavefront, n_workgroup, dev_wf_sz);
}

metrics kernel_launch_wmma_f32_16x16x16_f16(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz)
{
    //FP16 compute, FP32 accumulate
    return kernel_launch_wmma<rocwmma::float16_t,rocwmma::float32_t,16,16,16>(n_wavefront, n_workgroup, dev_wf_sz);
}

metrics kernel_launch_wmma_f32_16x16x16_bf16(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz)
{
    //BF16 compute, FP32 accumulate
    return kernel_launch_wmma<rocwmma::bfloat16_t,rocwmma::float32_t,16,16,16>(n_wavefront, n_workgroup, dev_wf_sz);
}

metrics kernel_launch_wmma_f32_16x16x8_xf32(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz)
{
    //TF32 compute, FP32 accumulate
    return kernel_launch_wmma<rocwmma::xfloat32_t,rocwmma::float32_t,16,16,8>(n_wavefront, n_workgroup, dev_wf_sz);
}

metrics kernel_launch_wmma_f64_16x16x4_f64(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz)
{
    //FP64 compute, FP64 accumulate
    return kernel_launch_wmma<rocwmma::float64_t,rocwmma::float64_t,16,16,4>(n_wavefront, n_workgroup, dev_wf_sz);
}

metrics kernel_launch_wmma_f32_16x16x4_f32(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz)
{
    //FP32 compute, FP32 accumulate
    return kernel_launch_wmma<rocwmma::float32_t,rocwmma::float32_t,16,16,4>(n_wavefront, n_workgroup, dev_wf_sz);
}

metrics kernel_launch_wmma_i32_16x16x32_i8(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz)
{
    //INT8 compute, INT32 accumulate
    return kernel_launch_wmma<rocwmma::int8_t,rocwmma::int32_t,16,16,32>(n_wavefront, n_workgroup, dev_wf_sz);
}



template<typename TMul, typename TAcc, int WMMA_M, int WMMA_N, int WMMA_K>
inline metrics kernel_launch_wmma(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz)
{
    // n_wavefront: number of wavefront (warp) per workgroup (block)
    //              this means number of thread per workgroup will be n_wavefront * dev_wf_sz
    TMul *dev_buf_A, *dev_buf_B; 
    TAcc *dev_buf_C;
    uint64_t *n_clockCount, *dev_n_clockCount;

    float gpu_elapsed_time_ms;
    hipEvent_t start;
    hipEvent_t stop;
    metrics run_metrics;

    // Creating Event Timer
    hipErrchk(hipEventCreate(&start));
    hipErrchk(hipEventCreate(&stop));

    // Memory Allocation
    // Array to store clock count per instruction; 
    // Threads in a wavefront executed in lock-step fashion, and thus we only record at wavefront granularity
    n_clockCount = (uint64_t*) malloc(n_wavefront * n_workgroup * sizeof(uint64_t));
    hipErrchk(hipMalloc((void**)&dev_n_clockCount, n_wavefront * n_workgroup * sizeof(uint64_t)));

    hipErrchk(hipMalloc((void**)&dev_buf_A, WMMA_M * WMMA_K * n_wavefront * n_workgroup * sizeof(TMul)));
    hipErrchk(hipMalloc((void**)&dev_buf_B, WMMA_K * WMMA_N * n_wavefront * n_workgroup * sizeof(TMul)));
    hipErrchk(hipMalloc((void**)&dev_buf_C, WMMA_M * WMMA_N * n_wavefront * n_workgroup * sizeof(TAcc)));

    // Launching Kernels
    hipErrchk(hipEventRecord(start,0));

    hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma<TMul, TAcc, WMMA_M, WMMA_N, WMMA_K>), dim3(n_workgroup), dim3(n_wavefront * dev_wf_sz), 0, 0,  dev_buf_A, dev_buf_B, dev_buf_C, dev_n_clockCount, dev_wf_sz);
    
    hipErrchk(hipEventRecord(stop,0)); 
    hipErrchk(hipDeviceSynchronize());

    // Calculate launch time
    hipErrchk(hipEventElapsedTime(&gpu_elapsed_time_ms,start,stop));

    // Processing n_clock
    hipErrchk(hipMemcpy(n_clockCount, dev_n_clockCount, n_wavefront * n_workgroup * sizeof(uint64_t), hipMemcpyDeviceToHost));
    
    //TODO: Calculate average, stdev, max, min of n_clock
    double average_clock, stdev_clock;
    uint64_t sum_clock = 0;
    uint64_t max_clock = n_clockCount[0];
    uint64_t min_clock = n_clockCount[0];
    for(int i = 0; i < n_wavefront * n_workgroup; i++)
    {
        sum_clock = (uint64_t) sum_clock + n_clockCount[i];
        if(n_clockCount[i] < min_clock) {min_clock = n_clockCount[i];}
        if(n_clockCount[i] > max_clock) {max_clock = n_clockCount[i];}
    }
    average_clock = (double) sum_clock / (n_wavefront * n_workgroup);
    
    double square_sum_clock = 0;
    for(int i = 0; i < n_wavefront * n_workgroup; i++)
    {
        square_sum_clock = square_sum_clock + pow((double) n_clockCount[i] / - average_clock, 2);
    }
    stdev_clock = sqrt(square_sum_clock / (n_wavefront * n_workgroup));

    // Clean-up Memory Allocation
    hipErrchk(hipEventDestroy(start));
    hipErrchk(hipEventDestroy(stop));

    hipErrchk(hipFree(dev_n_clockCount));
    free(n_clockCount);

    hipErrchk(hipFree(dev_buf_A));
    hipErrchk(hipFree(dev_buf_B));
    hipErrchk(hipFree(dev_buf_C));

    // Storing the run metrics
    run_metrics.n_iter      = NUM_LOOPS_WMMA;
    run_metrics.n_thread    = n_workgroup * n_wavefront * dev_wf_sz;
    run_metrics.n_wg        = n_workgroup;
    run_metrics.n_wf        = n_wavefront;
    run_metrics.wf_size     = dev_wf_sz;
    run_metrics.wg_size     = n_wavefront * dev_wf_sz;
    run_metrics.time_ms     = gpu_elapsed_time_ms;

    run_metrics.avg_clock   = average_clock;
    run_metrics.max_clock   = max_clock;
    run_metrics.min_clock   = min_clock;
    run_metrics.stdev_clock = stdev_clock;

    run_metrics.n_flops = (uint64_t) (2 * WMMA_M * WMMA_K * WMMA_N) * n_wavefront * n_workgroup * NUM_LOOPS_WMMA;
    run_metrics.n_bytes = (uint64_t) (2 * (WMMA_M*WMMA_K*sizeof(TMul) + WMMA_N*WMMA_K*sizeof(TMul) + WMMA_M*WMMA_N*sizeof(TAcc))) * n_wavefront * n_workgroup;

    run_metrics.gflop_per_s = (double) (run_metrics.n_flops / (run_metrics.time_ms*0.001))/1000000000;

    return run_metrics;
}