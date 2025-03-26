#pragma once
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <stdio.h>


// General ROCM error message

#define hipErrchk(ans)                    \
  {                                       \
    hipAssert((ans), __FILE__, __LINE__); \
  }


inline void
  hipAssert(hipError_t code, const char* file, int line, bool abort = true)
{
  if (code != hipSuccess)
  {
    fprintf(
      stderr, "[ERR!]: %s %s %d\n", hipGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// rocBLAS error message
static const char *rocBLASGetErrorString(rocblas_status error)
{
    switch (error)
    {
        case rocblas_status_success:
            return "ROCBLAS_STATUS_SUCCESS";

        case rocblas_status_invalid_value:
            return "ROCBLAS_STATUS_INVALID_VALUE";

        case rocblas_status_invalid_handle:
            return "ROCBLAS_STATUS_INVALID_HANDLE";

        case rocblas_status_not_implemented:
            return "ROCBLAS_STATUS_NOT_IMPLEMENTED";

        case rocblas_status_invalid_pointer:
            return "ROCBLAS_STATUS_INVALID_POINTER";

        case rocblas_status_invalid_size:
            return "ROCBLAS_STATUS_INVALID_SIZE";

        case rocblas_status_memory_error:
            return "ROCBLAS_STATUS_MEMORY_ERROR";

        case rocblas_status_internal_error:
            return "ROCBLAS_STATUS_INTERNAL_ERROR";

        case rocblas_status_perf_degraded:
            return "ROCBLAS_STATUS_PERFORMANCE_DEGRADED";

        case rocblas_status_size_query_mismatch:
            return "ROCBLAS_STATUS_QUERY_MISMATCH";

        case rocblas_status_size_increased:
            return "ROCBLAS_STATUS_SIZE_INCREASED";

        case rocblas_status_size_unchanged:
            return "ROCBLAS_STATUS_SIZE_UNCHANGED";

        case rocblas_status_continue:
            return "ROCBLAS_STATUS_CONTINUE";

        case rocblas_status_check_numerics_fail:
            return "ROCBLAS_STATUS_CHECK_NUMERICS_FAIL";

        case rocblas_status_excluded_from_build:
            return "ROCBLAS_STATUS_EXCLUDED_FROM_BUILD";

        case rocblas_status_arch_mismatch:
            return "ROCBLAS_STATUS_ARCHITECTURE_MISMATCH";
    }
    return "ROCBLAS_STATUS_UNKNOWN";
}

inline void
  hipAssert(rocblas_status code, const char* file, int line, bool abort = true)
{
  if (code != rocblas_status_success)
  {
    fprintf(
      stderr, "[ERR!]: %s %s %d\n", rocBLASGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


// hipBLAS error message
static const char *hipBLASGetErrorString(hipblasStatus_t error)
{
    switch (error)
    {
        case HIPBLAS_STATUS_SUCCESS:
            return "HIPBLAS_STATUS_SUCCESS";

        case HIPBLAS_STATUS_NOT_INITIALIZED:
            return "HIPBLAS_STATUS_NOT_INITIALIZED";

        case HIPBLAS_STATUS_ALLOC_FAILED:
            return "HIPBLAS_STATUS_ALLOC_FAILED";

        case HIPBLAS_STATUS_INVALID_VALUE:
            return "HIPBLAS_STATUS_INVALID_VALUE";

        case HIPBLAS_STATUS_MAPPING_ERROR:
            return "HIPBLAS_STATUS_MAPPING_ERROR";

        case HIPBLAS_STATUS_EXECUTION_FAILED:
            return "HIPBLAS_STATUS_EXECUTION_FAILED";

        case HIPBLAS_STATUS_INTERNAL_ERROR:
            return "HIPBLAS_STATUS_INTERNAL_ERROR";

        case HIPBLAS_STATUS_NOT_SUPPORTED:
            return "HIPBLAS_STATUS_NOT_SUPPORTED";

        case HIPBLAS_STATUS_ARCH_MISMATCH:
            return "HIPBLAS_STATUS_ARCH_MISMATCH";

        case HIPBLAS_STATUS_INVALID_ENUM:
            return "HIPBLAS_STATUS_INVALID_ENUM";

        case HIPBLAS_STATUS_UNKNOWN:
            return "HIPBLAS_STATUS_UNKNOWN";

        case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
            return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
    }
    return "HIPBLAS_STATUS_UNKNOWN";
}

// hipBLAS error message
inline void
  hipAssert(hipblasStatus_t code, const char* file, int line, bool abort = true)
{
  if (code != HIPBLAS_STATUS_SUCCESS)
  {
    fprintf(
      stderr, "[ERR!]: %s %s %d\n", hipBLASGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}