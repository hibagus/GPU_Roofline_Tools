#pragma once
#include <hip/hip_runtime.h>
#include <stdio.h>


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