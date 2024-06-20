#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <GPU_Roofline_Tools/utils/cuda/cudacheck.cuh>

template<typename T> inline void printcudaDeviceTable(T t, const int& width)
{
    std::stringstream ss;
    ss << t;
    std::cout  << std::left << std::setw(width) << std::setfill(' ') << ss.str().substr(0,width);
}

inline void print_cuda_device_info(int nDevices)
{
    std::cout << "[INFO] Detected " << nDevices << " CUDA-capable device(s)\n";
    std::cout << "[INFO] +---+------------------------+----+----+-----------+--------------+--------------+\n"; 
    printcudaDeviceTable("[INFO] |", 8); 
    printcudaDeviceTable("#", 3);                  printcudaDeviceTable("|", 1);
    printcudaDeviceTable("Device Name", 24);       printcudaDeviceTable("|", 1);
    printcudaDeviceTable("CC", 4);      printcudaDeviceTable("|", 1);
    printcudaDeviceTable("#SM", 4);       printcudaDeviceTable("|", 1);
    printcudaDeviceTable("Freq. (MHz)", 11);     printcudaDeviceTable("|", 1);
    printcudaDeviceTable("Mem. Size (MB)", 14);  printcudaDeviceTable("|", 1);
    printcudaDeviceTable("Mem. BW (GB/s)", 14);  printcudaDeviceTable("|", 1);
    printcudaDeviceTable("\n", 1);
    std::cout << "[INFO] +---+------------------------+----+----+-----------+--------------+--------------+\n";  
    
    for (int i = 0; i < nDevices; i++) 
    {
        cudaDeviceProp prop;
        gpuErrchk(cudaGetDeviceProperties(&prop, i));
        printcudaDeviceTable("[INFO] |", 8); 
        printcudaDeviceTable(i, 3);                  printcudaDeviceTable("|", 1);
        printcudaDeviceTable(prop.name, 24);       printcudaDeviceTable("|", 1);
        printcudaDeviceTable(std::to_string(prop.major)+"."+std::to_string(prop.minor), 4);      printcudaDeviceTable("|", 1);
        printcudaDeviceTable(prop.multiProcessorCount, 4);   printcudaDeviceTable("|", 1);
        printcudaDeviceTable(prop.clockRate/1000, 11); printcudaDeviceTable("|", 1);
        printcudaDeviceTable(prop.totalGlobalMem/1048576, 14);  printcudaDeviceTable("|", 1);
        printcudaDeviceTable(2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6, 14);  printcudaDeviceTable("|", 1);
        printcudaDeviceTable("\n", 1);
        std::cout << "[INFO] +---+------------------------+----+----+-----------+--------------+--------------+\n";  
    }
}

inline void print_no_cuda_devices()
{
    std::cerr << "---------------------------------------------------------------"
           "--------------------\n";
    std::cerr << "---------------------------------------------------------------"
                 "--------------------\n";
    std::cerr << "[ERR!]: No CUDA-capable devices are detected. Program will now exit.\n";
    std::cerr << "       Please check whether your system has CUDA-capable device installed"
                 " and the CUDA driver is installed correctly.\n";       
    std::cerr << "---------------------------------------------------------------"
                 "--------------------\n";
    std::cerr << "---------------------------------------------------------------"
                 "--------------------\n";
    exit(1);
}