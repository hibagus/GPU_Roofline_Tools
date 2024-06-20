#include <hip/hip_runtime.h>
#include <GPU_Roofline_Tools/kernel/rocm/add.hip.h>
#include <GPU_Roofline_Tools/launcher/rocm/kernel_launch.hip.h>
#include <GPU_Roofline_Tools/utils/rocm/hipcheck.hip.h>
#include <GPU_Roofline_Tools/utils/rocm/hipinfo.hip.h>


int kernel_launch_fp64(uint32_t n_thr_per_wg, uint32_t n_wg)
{
    double *dev_buf_A, *dev_buf_B;
    uint32_t size = n_thr_per_wg * n_wg;

    int nDevices;
    hipErrchk(hipGetDeviceCount(&nDevices));
    print_hip_device_info(nDevices);
    if(nDevices>0) {std::cout << "[WARN] This program does not currently support Multi-GPU run.\n";}

    hipErrchk(hipMalloc((void**)&dev_buf_A, size * sizeof(double)));
    hipErrchk(hipMalloc((void**)&dev_buf_B, size * sizeof(double)));
 
    hipLaunchKernelGGL(add<double>, dim3(n_wg), dim3(n_thr_per_wg), 0, 0,  dev_buf_A, dev_buf_B, n_thr_per_wg, n_wg);

    hipErrchk(hipDeviceSynchronize());

    hipErrchk(hipFree(dev_buf_A));
    hipErrchk(hipFree(dev_buf_B));
    printf("Finished running kernel\n");
    return 0;
}