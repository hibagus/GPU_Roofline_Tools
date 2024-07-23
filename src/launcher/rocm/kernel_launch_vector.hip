#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_fp16.h>
#include <GPU_Roofline_Tools/kernel/rocm/add.hip.h>
#include <GPU_Roofline_Tools/kernel/rocm/mul.hip.h>
#include <GPU_Roofline_Tools/kernel/rocm/fma.hip.h>
#include <GPU_Roofline_Tools/launcher/rocm/kernel_launch_vector.hip.h>
#include <GPU_Roofline_Tools/utils/rocm/hipcheck.hip.h>
#include <GPU_Roofline_Tools/utils/rocm/hipinfo.hip.h>
#include <GPU_Roofline_Tools/utils/common/optype.h>
#include <GPU_Roofline_Tools/utils/common/metrics.h>

metrics kernel_launch_vector_bf16(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t n_loop, optype op)
{
    return kernel_launch_vector<hip_bfloat16>(n_wavefront, n_workgroup, n_loop, op);
}

metrics kernel_launch_vector_fp16(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t n_loop, optype op)
{
    return kernel_launch_vector<_Float16>(n_wavefront, n_workgroup, n_loop, op);
}

metrics kernel_launch_vector_fp32(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t n_loop, optype op)
{
    return kernel_launch_vector<float>(n_wavefront, n_workgroup, n_loop, op);
}

metrics kernel_launch_vector_fp64(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t n_loop, optype op)
{
    return kernel_launch_vector<double>(n_wavefront, n_workgroup, n_loop, op);
}

template<typename TCompute>
inline metrics kernel_launch_vector(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t n_loop, optype op)
{
    // n_wavefront: number of wavefront (warp) per workgroup (block)
    //              this means number of thread per workgroup will be n_wavefront * wf_sz
    TCompute *dev_buf_A, *dev_buf_B, *dev_buf_C;
    const TCompute constant = (TCompute) 1.00;
    uint64_t *n_clockCount, *dev_n_clockCount;

    uint32_t p_sz;  // Problem Size (total threads)
    uint32_t wf_sz; // Wavefront Size; same as warpSize in CUDA (number of threads per wavefront)
    uint32_t wg_sz; // Workgroup Size; same as blockSize in CUDA (number of threads per workgroup)
    uint32_t max_wg_sz; // Max Workgroup Size; (max. number of threads per workgroup)

    float gpu_elapsed_time_ms;
    hipEvent_t start;
    hipEvent_t stop;
    metrics run_metrics;


    // Detect HIP-capable Devices
    int nDevices;
    hipErrchk(hipGetDeviceCount(&nDevices));
    if(nDevices>0)    {print_hip_device_info(nDevices); std::cout << "[WARN] This program does not currently support Multi-GPU run.\n";}
    else              {print_no_hip_devices();}

    // Creating Event Timer
    hipErrchk(hipEventCreate(&start));
    hipErrchk(hipEventCreate(&stop));

    // Get to know the device
    int device;
    hipErrchk(hipGetDevice(&device));
    hipDeviceProp_t prop;
    hipErrchk(hipGetDeviceProperties(&prop, device));
    wf_sz     = prop.warpSize;
    wg_sz     = n_wavefront * wf_sz;
    max_wg_sz = prop.maxThreadsPerBlock;

    if(wg_sz > max_wg_sz)
    {
        std::cerr << "[ERR!]: Number of wavefront per workgroup results in total thread per workgroup exceeding the maximum value.\n";
        exit(1);
    }


    // Calculate problem size (basically number of threads)
    p_sz = n_wavefront * wf_sz * n_workgroup;

    // Memory Allocation
    // Array to store clock count per instruction; 
    // Threads in a wavefront executed in lock-step fashion, and thus we only record at wavefront granularity
    n_clockCount = (uint64_t*) malloc(n_wavefront * n_workgroup * sizeof(uint64_t));
    hipErrchk(hipMalloc((void**)&dev_n_clockCount, n_wavefront * n_workgroup * sizeof(uint64_t)));

    
    if (op == V_MUL || op == V_FMA1)
    {
        hipErrchk(hipMalloc((void**)&dev_buf_A, p_sz * sizeof(TCompute)));
    }
    else if (op == V_ADD || op == V_FMA2)
    {
        hipErrchk(hipMalloc((void**)&dev_buf_A, p_sz * sizeof(TCompute)));
        hipErrchk(hipMalloc((void**)&dev_buf_B, p_sz * sizeof(TCompute)));
    }
    else if (op == V_FMA3)
    {
        hipErrchk(hipMalloc((void**)&dev_buf_A, p_sz * sizeof(TCompute)));
        hipErrchk(hipMalloc((void**)&dev_buf_B, p_sz * sizeof(TCompute)));
        hipErrchk(hipMalloc((void**)&dev_buf_C, p_sz * sizeof(TCompute)));
    }

    // Launching Kernels
    hipErrchk(hipEventRecord(start,0));
    if      (op == V_MUL)   {hipLaunchKernelGGL(mul<TCompute>, dim3(n_workgroup), dim3(n_wavefront * wf_sz), 0, 0,  dev_buf_A, constant, n_loop, dev_n_clockCount, wf_sz);}
    else if (op == V_FMA1)  {hipLaunchKernelGGL(fma<TCompute>, dim3(n_workgroup), dim3(n_wavefront * wf_sz), 0, 0,  dev_buf_A, constant, n_loop, dev_n_clockCount, wf_sz);}
    else if (op == V_ADD)   {hipLaunchKernelGGL(add<TCompute>, dim3(n_workgroup), dim3(n_wavefront * wf_sz), 0, 0,  dev_buf_A, dev_buf_B, n_loop, dev_n_clockCount, wf_sz);}
    else if (op == V_FMA2)  {hipLaunchKernelGGL(fma<TCompute>, dim3(n_workgroup), dim3(n_wavefront * wf_sz), 0, 0,  dev_buf_A, constant, dev_buf_B, n_loop, dev_n_clockCount, wf_sz);}
    else if (op == V_FMA3)  {hipLaunchKernelGGL(fma<TCompute>, dim3(n_workgroup), dim3(n_wavefront * wf_sz), 0, 0,  dev_buf_A, dev_buf_B, dev_buf_C, n_loop, dev_n_clockCount, wf_sz);}
    hipErrchk(hipEventRecord(stop,0));
    hipErrchk(hipDeviceSynchronize());

    // Calculate launch time
    hipEventElapsedTime(&gpu_elapsed_time_ms,start,stop);


    // Processing n_clock
    hipErrchk(hipMemcpy(n_clockCount, dev_n_clockCount, n_wavefront * n_workgroup * sizeof(uint64_t), hipMemcpyDeviceToHost));
    
    //TODO: Calculate average, stdev, max, min of n_clock
    double average_clock, stdev_clock;
    uint64_t sum_clock = 0;
    uint64_t max_clock = n_clockCount[0];
    uint64_t min_clock = n_clockCount[0];
    for(int i = 0; i < n_wavefront * n_workgroup; i++)
    {
        sum_clock = sum_clock + n_clockCount[i];
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

    if (op == V_MUL || op == V_FMA1)
    {
        hipErrchk(hipFree(dev_buf_A));
    }
    else if (op == V_ADD || op == V_FMA2)
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
    run_metrics.n_iter      = n_loop;
    run_metrics.n_thread    = p_sz;
    run_metrics.n_wg        = n_workgroup;
    run_metrics.n_wf        = n_wavefront;
    run_metrics.wf_size     = wf_sz;
    run_metrics.wg_size     = wg_sz;
    run_metrics.time_ms     = gpu_elapsed_time_ms;

    run_metrics.avg_clock   = average_clock;
    run_metrics.max_clock   = max_clock;
    run_metrics.min_clock   = min_clock;
    run_metrics.stdev_clock = stdev_clock;

    if      (op == V_MUL)   {run_metrics.n_flops = 1*p_sz*n_loop; run_metrics.n_bytes = 1*p_sz*sizeof(TCompute)*n_loop;}
    else if (op == V_FMA1)  {run_metrics.n_flops = 2*p_sz*n_loop; run_metrics.n_bytes = 1*p_sz*sizeof(TCompute)*n_loop;}
    else if (op == V_ADD)   {run_metrics.n_flops = 1*p_sz*n_loop; run_metrics.n_bytes = 2*p_sz*sizeof(TCompute)*n_loop;}
    else if (op == V_FMA2)  {run_metrics.n_flops = 2*p_sz*n_loop; run_metrics.n_bytes = 2*p_sz*sizeof(TCompute)*n_loop;}
    else if (op == V_FMA3)  {run_metrics.n_flops = 2*p_sz*n_loop; run_metrics.n_bytes = 3*p_sz*sizeof(TCompute)*n_loop;}

    printf("Finished running kernel\n");
    return run_metrics;
}