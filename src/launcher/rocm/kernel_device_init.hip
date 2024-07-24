#include <iostream>
#include <hip/hip_runtime.h>
#include <GPU_Roofline_Tools/launcher/rocm/kernel_device_init.hip.h>
#include <GPU_Roofline_Tools/utils/rocm/hipcheck.hip.h>
#include <GPU_Roofline_Tools/utils/rocm/hipinfo.hip.h>

void initHIPDevice()
{
    // Detect HIP-capable Devices
    int nDevices;
    hipErrchk(hipGetDeviceCount(&nDevices));
    if(nDevices>0)    {print_hip_device_info(nDevices); std::cout << "[WARN] This program does not currently support Multi-GPU run.\n";}
    else              {print_no_hip_devices();}
}

void setHIPDevice(uint32_t device_index)
{
    int nDevices;
    hipErrchk(hipGetDeviceCount(&nDevices));

    if(device_index<nDevices)
    {
        hipErrchk(hipSetDevice(device_index));
        std::cout << "[INFO] Set default GPU to GPU index " << device_index << std::endl;
    }
    else
    {
        std::cerr << "[ERR!] GPU index out of bound!" << std::endl;
        exit(1);
    }
    
}

uint32_t getMaxWorkgroupSize(uint32_t device_index)
{
    // Get to know the device
    hipDeviceProp_t prop;
    hipErrchk(hipGetDeviceProperties(&prop, device_index));
    return prop.maxThreadsPerBlock;
}

uint32_t getWaveFrontSize(uint32_t device_index)
{
    hipDeviceProp_t prop;
    hipErrchk(hipGetDeviceProperties(&prop, device_index));
    return prop.warpSize;
}