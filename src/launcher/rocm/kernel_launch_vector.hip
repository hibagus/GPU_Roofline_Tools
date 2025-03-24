#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_fp16.h>
#include <GPU_Roofline_Tools/kernel/rocm/add.hip.h>
#include <GPU_Roofline_Tools/kernel/rocm/mul.hip.h>
#include <GPU_Roofline_Tools/kernel/rocm/fma.hip.h>
#include <GPU_Roofline_Tools/launcher/rocm/kernel_launch_vector.hip.h>
#include <GPU_Roofline_Tools/utils/rocm/hipcheck.hip.h>
#include <GPU_Roofline_Tools/utils/common/optype.h>
#include <GPU_Roofline_Tools/utils/common/metrics.h>
#include <GPU_Roofline_Tools/utils/common/global.h>

metrics kernel_launch_vector_bf16(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz, optype op)
{
    return kernel_launch_vector<hip_bfloat16>(n_wavefront, n_workgroup, dev_wf_sz, op);
}

metrics kernel_launch_vector_fp16(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz, optype op)
{
    return kernel_launch_vector<half>(n_wavefront, n_workgroup, dev_wf_sz, op);
}

metrics kernel_launch_vector_fp32(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz, optype op)
{
    return kernel_launch_vector<float>(n_wavefront, n_workgroup, dev_wf_sz, op);
}

metrics kernel_launch_vector_fp64(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz, optype op)
{
    return kernel_launch_vector<double>(n_wavefront, n_workgroup, dev_wf_sz, op);
}

template<typename TCompute>
inline metrics kernel_launch_vector(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz, optype op)
{
    // n_wavefront: number of wavefront (warp) per workgroup (block)
    //              this means number of thread per workgroup will be n_wavefront * dev_wf_sz
    TCompute *dev_buf_A, *dev_buf_B, *dev_buf_C;
    const TCompute constant = (TCompute) 0.000;
    uint64_t *n_clockCount, *dev_n_clockCount;

    // Calculate problem size (basically number of threads)
    uint32_t problem_size = n_wavefront * dev_wf_sz * n_workgroup;

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

    if (op == V_MUL1 || op == V_FMA1 || op == V_ADD1)
    {
        hipErrchk(hipMalloc((void**)&dev_buf_A, problem_size * sizeof(TCompute)));
    }
    else if (op == V_ADD2 || op == V_FMA2 || op == V_MUL2)
    {
        hipErrchk(hipMalloc((void**)&dev_buf_A, problem_size * sizeof(TCompute)));
        hipErrchk(hipMalloc((void**)&dev_buf_B, problem_size * sizeof(TCompute)));
    }
    else if (op == V_FMA3)
    {
        hipErrchk(hipMalloc((void**)&dev_buf_A, problem_size * sizeof(TCompute)));
        hipErrchk(hipMalloc((void**)&dev_buf_B, problem_size * sizeof(TCompute)));
        hipErrchk(hipMalloc((void**)&dev_buf_C, problem_size * sizeof(TCompute)));
    }

    // Launching Kernels
    hipErrchk(hipEventRecord(start,0));
    if      (op == V_MUL1)  {hipLaunchKernelGGL(HIP_KERNEL_NAME(mul<TCompute>), dim3(n_workgroup), dim3(n_wavefront * dev_wf_sz), 0, 0,  dev_buf_A, constant, dev_n_clockCount, dev_wf_sz);}
    else if (op == V_ADD1)  {hipLaunchKernelGGL(HIP_KERNEL_NAME(add<TCompute>), dim3(n_workgroup), dim3(n_wavefront * dev_wf_sz), 0, 0,  dev_buf_A, constant, dev_n_clockCount, dev_wf_sz);}
    else if (op == V_FMA1)  {hipLaunchKernelGGL(HIP_KERNEL_NAME(fma<TCompute>), dim3(n_workgroup), dim3(n_wavefront * dev_wf_sz), 0, 0,  dev_buf_A, constant, dev_n_clockCount, dev_wf_sz);}
    else if (op == V_ADD2)  {hipLaunchKernelGGL(HIP_KERNEL_NAME(add<TCompute>), dim3(n_workgroup), dim3(n_wavefront * dev_wf_sz), 0, 0,  dev_buf_A, dev_buf_B, dev_n_clockCount, dev_wf_sz);}
    else if (op == V_MUL2)  {hipLaunchKernelGGL(HIP_KERNEL_NAME(mul<TCompute>), dim3(n_workgroup), dim3(n_wavefront * dev_wf_sz), 0, 0,  dev_buf_A, dev_buf_B, dev_n_clockCount, dev_wf_sz);}
    //else if (op == V_ADD2)  {hipLaunchKernelGGL(HIP_KERNEL_NAME(add<TCompute>), dim3(n_workgroup), dim3(n_wavefront * dev_wf_sz), 0, 0,  dev_buf_A, dev_buf_B, dev_buf_C, dev_n_clockCount, dev_wf_sz);}
    else if (op == V_FMA2)  {hipLaunchKernelGGL(HIP_KERNEL_NAME(fma<TCompute>), dim3(n_workgroup), dim3(n_wavefront * dev_wf_sz), 0, 0,  dev_buf_A, constant, dev_buf_B, dev_n_clockCount, dev_wf_sz);}
    else if (op == V_FMA3)  {hipLaunchKernelGGL(HIP_KERNEL_NAME(fma<TCompute>), dim3(n_workgroup), dim3(n_wavefront * dev_wf_sz), 0, 0,  dev_buf_A, dev_buf_B, dev_buf_C, dev_n_clockCount, dev_wf_sz);}
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

    if (op == V_MUL1 || op == V_FMA1 || op == V_ADD1)
    {
        hipErrchk(hipFree(dev_buf_A));
    }
    else if (op == V_ADD2 || op == V_FMA2 || op == V_MUL2)
    {
        hipErrchk(hipFree(dev_buf_A));
        hipErrchk(hipFree(dev_buf_B));
    }
    else if (op == V_FMA3)
    {
        hipErrchk(hipFree(dev_buf_A));
        hipErrchk(hipFree(dev_buf_B));
        hipErrchk(hipFree(dev_buf_C));
    }

    // Storing the run metrics
    run_metrics.n_iter      = NUM_LOOPS;
    run_metrics.n_thread    = problem_size;
    run_metrics.n_wg        = n_workgroup;
    run_metrics.n_wf        = n_wavefront;
    run_metrics.wf_size     = dev_wf_sz;
    run_metrics.wg_size     = n_wavefront * dev_wf_sz;
    run_metrics.time_ms     = gpu_elapsed_time_ms;

    run_metrics.avg_clock   = average_clock;
    run_metrics.max_clock   = max_clock;
    run_metrics.min_clock   = min_clock;
    run_metrics.stdev_clock = stdev_clock;

    if      (op == V_MUL1)  {run_metrics.n_flops = (uint64_t) 1*problem_size*NUM_LOOPS; run_metrics.n_bytes = (uint64_t) 1*problem_size*sizeof(TCompute);}
    else if (op == V_ADD1)  {run_metrics.n_flops = (uint64_t) 1*problem_size*NUM_LOOPS; run_metrics.n_bytes = (uint64_t) 1*problem_size*sizeof(TCompute);}
    else if (op == V_FMA1)  {run_metrics.n_flops = (uint64_t) 2*problem_size*NUM_LOOPS; run_metrics.n_bytes = (uint64_t) 1*problem_size*sizeof(TCompute);}
    else if (op == V_ADD2)  {run_metrics.n_flops = (uint64_t) 1*problem_size*NUM_LOOPS; run_metrics.n_bytes = (uint64_t) 2*problem_size*sizeof(TCompute);}
    else if (op == V_MUL2)  {run_metrics.n_flops = (uint64_t) 1*problem_size*NUM_LOOPS; run_metrics.n_bytes = (uint64_t) 2*problem_size*sizeof(TCompute);}
    //else if (op == V_ADD2)  {run_metrics.n_flops = (uint64_t) 1*problem_size*NUM_LOOPS; run_metrics.n_bytes = (uint64_t) 3*problem_size*sizeof(TCompute);}
    else if (op == V_FMA2)  {run_metrics.n_flops = (uint64_t) 2*problem_size*NUM_LOOPS; run_metrics.n_bytes = (uint64_t) 2*problem_size*sizeof(TCompute);}
    else if (op == V_FMA3)  {run_metrics.n_flops = (uint64_t) 2*problem_size*NUM_LOOPS; run_metrics.n_bytes = (uint64_t) 3*problem_size*sizeof(TCompute);}
    
    run_metrics.gflop_per_s = (double) (run_metrics.n_flops / (run_metrics.time_ms*0.001))/1000000000;
    return run_metrics;
}
