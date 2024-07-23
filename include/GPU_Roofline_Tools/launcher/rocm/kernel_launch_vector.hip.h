#pragma once
#include <GPU_Roofline_Tools/utils/common/optype.h>
#include <GPU_Roofline_Tools/utils/common/metrics.h>

template<typename TCompute>
inline metrics kernel_launch_vector(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t n_loop, optype op);

metrics kernel_launch_vector_fp16(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t n_loop, optype op);
metrics kernel_launch_vector_fp32(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t n_loop, optype op);
metrics kernel_launch_vector_fp64(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t n_loop, optype op);