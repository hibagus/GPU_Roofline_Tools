#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <GPU_Roofline_Tools/utils/rocm/hipinfo.hip.h>

template<typename T> inline void printhipDeviceTable(T t, const int& width)
{
    std::stringstream ss;
    ss << t;
    std::cout  << std::left << std::setw(width) << std::setfill(' ') << ss.str().substr(0,width);
}

inline void print_hip_device_info(int nDevices)
{
    std::cout << "[INFO] Detected " << nDevices << " HIP-capable device(s)\n";
    std::cout << "[INFO] +---+------------------------+----+----+-----------+--------------+--------------+\n"; 
    printhipDeviceTable("[INFO] |", 8); 
    printhipDeviceTable("#", 3);                  printhipDeviceTable("|", 1);
    printhipDeviceTable("Device Name", 24);       printhipDeviceTable("|", 1);
    printhipDeviceTable("CC", 4);      printhipDeviceTable("|", 1);
    printhipDeviceTable("#CU", 4);       printhipDeviceTable("|", 1);
    printhipDeviceTable("Freq. (MHz)", 11);     printhipDeviceTable("|", 1);
    printhipDeviceTable("Mem. Size (MB)", 14);  printhipDeviceTable("|", 1);
    printhipDeviceTable("Mem. BW (GB/s)", 14);  printhipDeviceTable("|", 1);
    printhipDeviceTable("\n", 1);
    std::cout << "[INFO] +---+------------------------+----+----+-----------+--------------+--------------+\n";  
    
    for (int i = 0; i < nDevices; i++) 
    {
        hipDeviceProp_t prop;
        hipErrchk(hipGetDeviceProperties(&prop, i));
        printhipDeviceTable("[INFO] |", 8); 
        printhipDeviceTable(i, 3);                  printhipDeviceTable("|", 1);
        printhipDeviceTable(prop.name, 24);       printhipDeviceTable("|", 1);
        printhipDeviceTable(std::to_string(prop.major)+"."+std::to_string(prop.minor), 4);      printhipDeviceTable("|", 1);
        printhipDeviceTable(prop.multiProcessorCount, 4);   printhipDeviceTable("|", 1);
        printhipDeviceTable(prop.clockRate/1000, 11); printhipDeviceTable("|", 1);
        printhipDeviceTable(prop.totalGlobalMem/1048576, 14);  printhipDeviceTable("|", 1);
        printhipDeviceTable(2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6, 14);  printhipDeviceTable("|", 1);
        printhipDeviceTable("\n", 1);
        std::cout << "[INFO] +---+------------------------+----+----+-----------+--------------+--------------+\n";  
    }
}

inline void print_no_hip_devices()
{
    std::cerr << "---------------------------------------------------------------"
           "--------------------\n";
    std::cerr << "---------------------------------------------------------------"
                 "--------------------\n";
    std::cerr << "[ERR!]: No HIP-capable devices are detected. Program will now exit.\n";
    std::cerr << "       Please check whether your system has HIP-capable device installed"
                 " and the HIP driver is installed correctly.\n";       
    std::cerr << "---------------------------------------------------------------"
                 "--------------------\n";
    std::cerr << "---------------------------------------------------------------"
                 "--------------------\n";
    exit(1);
}