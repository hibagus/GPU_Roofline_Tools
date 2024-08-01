#pragma once
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
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