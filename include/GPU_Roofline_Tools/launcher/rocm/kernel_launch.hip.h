#pragma once
#include <GPU_Roofline_Tools/utils/common/optype.h>

template<typename TCompute>
inline int kernel_launch(uint32_t n_thr_per_wg, uint32_t n_wg, optype op);

int kernel_launch_fp16(uint32_t n_thr_per_wg, uint32_t n_wg, optype op);
int kernel_launch_fp32(uint32_t n_thr_per_wg, uint32_t n_wg, optype op);
int kernel_launch_fp64(uint32_t n_thr_per_wg, uint32_t n_wg, optype op);