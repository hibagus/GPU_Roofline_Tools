#pragma once
#include <GPU_Roofline_Tools/utils/common/optype.h>
#include <GPU_Roofline_Tools/utils/common/metrics.h>

template<typename TMul, typename TAcc, int WMMA_M, int WMMA_N, int WMMA_K>
inline metrics kernel_launch_wmma(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz);


metrics kernel_launch_wmma_f32_16x16x32_fp8_fp8(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz);

metrics kernel_launch_wmma_f32_16x16x32_bf8_bf8(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz);

metrics kernel_launch_wmma_f32_16x16x16_f16(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz);

metrics kernel_launch_wmma_f32_16x16x16_bf16(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz);

metrics kernel_launch_wmma_f32_16x16x8_xf32(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz);

metrics kernel_launch_wmma_f64_16x16x4_f64(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz);

metrics kernel_launch_wmma_f32_16x16x4_f32(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz);

metrics kernel_launch_wmma_i32_16x16x32_i8(uint32_t n_wavefront, uint32_t n_workgroup, uint32_t dev_wf_sz);