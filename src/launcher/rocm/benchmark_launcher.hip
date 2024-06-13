#include <hip/hip_runtime.h>
#include <GPU_Roofline_Tools/kernel/rocm/add.hpp>
#include <GPU_Roofline_Tools/launcher/benchmark_launcher.hpp>

int launch_add()
{
    double *dev_buf_A, *dev_buf_B, *dev_buf_C;
    uint32_t num_thread_per_workgroup = 1024;
    uint32_t num_workgroup            = 1;
    uint32_t size = num_thread_per_workgroup * num_workgroup;

    hipMalloc((void**)&dev_buf_A, size * sizeof(double));
    hipMalloc((void**)&dev_buf_B, size * sizeof(double));
    hipMalloc((void**)&dev_buf_C, size * sizeof(double));
 
    hipLaunchKernelGGL(add_benchmark_memory, dim3(num_workgroup), dim3(num_thread_per_workgroup), 0, 0,  dev_buf_A, dev_buf_B, dev_buf_C, 1024, 1);

    hipFree(dev_buf_A);
    hipFree(dev_buf_B);
    hipFree(dev_buf_C);
    return 0;
}