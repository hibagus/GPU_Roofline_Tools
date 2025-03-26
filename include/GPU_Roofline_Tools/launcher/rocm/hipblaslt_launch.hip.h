#pragma once
#include <GPU_Roofline_Tools/utils/common/ptype.h>
#include <GPU_Roofline_Tools/utils/common/metrics.h>

template<typename TMul, typename TAcc, typename TScale>
inline metrics hipblaslt_launch(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, ptype mulOp, ptype accOp, ptype scaleOp, bool use_workspace);


metrics hipblaslt_launch_fp8_fp8_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace);

metrics hipblaslt_launch_fp8_fp16_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace);

metrics hipblaslt_launch_fp8_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace);

metrics hipblaslt_launch_bf8_bf8_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace);

metrics hipblaslt_launch_bf8_fp16_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace);

metrics hipblaslt_launch_bf8_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace);

metrics hipblaslt_launch_fp16_fp16_fp16(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace);

metrics hipblaslt_launch_bf16_bf16_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace);

metrics hipblaslt_launch_bf16_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace);

metrics hipblaslt_launch_fp16_fp16_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace);

metrics hipblaslt_launch_fp16_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace);

metrics hipblaslt_launch_tf32_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace);

metrics hipblaslt_launch_fp32_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace);

metrics hipblaslt_launch_fp64_fp64_fp64(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace);

metrics hipblaslt_launch_int8_int32_int32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace);