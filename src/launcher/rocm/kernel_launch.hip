#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_fp16.h>
#include <GPU_Roofline_Tools/kernel/rocm/add.hip.h>
#include <GPU_Roofline_Tools/launcher/rocm/kernel_launch.hip.h>
#include <GPU_Roofline_Tools/utils/rocm/hipcheck.hip.h>
#include <GPU_Roofline_Tools/utils/rocm/hipinfo.hip.h>
#include <GPU_Roofline_Tools/utils/common/optype.h>
#include <GPU_Roofline_Tools/utils/common/metrics.h>

metrics kernel_launch_bf16(uint32_t n_thr_per_wg, uint32_t n_wg, optype op)
{
    return kernel_launch<hip_bfloat16>(n_thr_per_wg, n_wg, op);
}

metrics kernel_launch_fp16(uint32_t n_thr_per_wg, uint32_t n_wg, optype op)
{
    return kernel_launch<_Float16>(n_thr_per_wg, n_wg, op);
}

metrics kernel_launch_fp32(uint32_t n_thr_per_wg, uint32_t n_wg, optype op)
{
    return kernel_launch<float>(n_thr_per_wg, n_wg, op);
}

metrics kernel_launch_fp64(uint32_t n_thr_per_wg, uint32_t n_wg, optype op)
{
    return kernel_launch<double>(n_thr_per_wg, n_wg, op);
}

template<typename TCompute>
inline metrics kernel_launch(uint32_t n_thr_per_wg, uint32_t n_wg, optype op)
{
    TCompute *dev_buf_A, *dev_buf_B, *dev_buf_C;
    uint32_t size = n_thr_per_wg * n_wg;
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


    // Memory Allocation
    if (op == V_MUL || op == V_FMA1)
    {
        hipErrchk(hipMalloc((void**)&dev_buf_A, size * sizeof(TCompute)));
    }
    else if (op == V_ADD || op == V_FMA2)
    {
        hipErrchk(hipMalloc((void**)&dev_buf_A, size * sizeof(TCompute)));
        hipErrchk(hipMalloc((void**)&dev_buf_B, size * sizeof(TCompute)));
    }
    else if (op == V_FMA3)
    {
        hipErrchk(hipMalloc((void**)&dev_buf_A, size * sizeof(TCompute)));
        hipErrchk(hipMalloc((void**)&dev_buf_B, size * sizeof(TCompute)));
        hipErrchk(hipMalloc((void**)&dev_buf_C, size * sizeof(TCompute)));
    }

    // Launching Kernels
    hipErrchk(hipEventRecord(start,0));
    hipLaunchKernelGGL(add<TCompute>, dim3(n_wg), dim3(n_thr_per_wg), 0, 0,  dev_buf_A, dev_buf_B, n_thr_per_wg, n_wg);
    hipErrchk(hipEventRecord(stop,0));
    hipErrchk(hipDeviceSynchronize());

    // Calculate launch time
    hipEventElapsedTime(&gpu_elapsed_time_ms,start,stop);


    // Clean-up Memory Allocation
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

    printf("Finished running kernel\n");
    return run_metrics;
}