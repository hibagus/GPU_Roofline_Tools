#pragma once
#include <GPU_Roofline_Tools/utils/common/optype.h>
#include <GPU_Roofline_Tools/utils/common/metrics.h>

template<typename TCompute>
inline metrics kernel_launch(uint32_t n_thr_per_wg, uint32_t n_wg, uint32_t n_loop, optype op);

metrics kernel_launch_fp16(uint32_t n_thr_per_wg, uint32_t n_wg, uint32_t n_loop, optype op);
metrics kernel_launch_fp32(uint32_t n_thr_per_wg, uint32_t n_wg, uint32_t n_loop, optype op);
metrics kernel_launch_fp64(uint32_t n_thr_per_wg, uint32_t n_wg, uint32_t n_loop, optype op);