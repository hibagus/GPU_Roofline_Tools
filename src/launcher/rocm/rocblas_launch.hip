#define ROCBLAS_BETA_FEATURES_API 1
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_fp16.h>
#include <rocblas/rocblas.h>
#include <rocblas/internal/rocblas_xfloat32.h>
#include <rocblas/internal/rocblas_float8.h>
#include <GPU_Roofline_Tools/launcher/rocm/rocblas_launch.hip.h>
#include <GPU_Roofline_Tools/utils/rocm/hipcheck.hip.h>
#include <GPU_Roofline_Tools/utils/common/ptype.h>
#include <GPU_Roofline_Tools/utils/common/metrics.h>
#include <GPU_Roofline_Tools/utils/common/global.h>


metrics rocblas_launch_fp8_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz)
{
    //FP8 multiply, FP32 accumulate, FP32 compute
    return rocblas_launch_float8_beta<rocblas_f8,float,float>(dim_M, dim_N, dim_K, dev_wf_sz, FP8, FP32, FP32);
}

metrics rocblas_launch_bf8_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz)
{
    //BF8 multiply, FP32 accumulate, FP32 compute
    return rocblas_launch_float8_beta<rocblas_bf8,float,float>(dim_M, dim_N, dim_K, dev_wf_sz, FP8, FP32, FP32);
}

metrics rocblas_launch_fp16_fp16_fp16(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz)
{
    //FP16 multiply, FP16 accumulate, FP16 compute
    //This does not run on Matrix Cores of MI300X
    return rocblas_launch<half,half,half>(dim_M, dim_N, dim_K, dev_wf_sz, FP16, FP16, FP16);
}

metrics rocblas_launch_fp16_fp16_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz)
{
    //FP16 multiply, FP16 accumulate, FP32 compute
    return rocblas_launch<half,half,float>(dim_M, dim_N, dim_K, dev_wf_sz, FP16, FP16, FP32);
}

metrics rocblas_launch_fp16_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz)
{
    //FP16 multiply, FP32 accumulate, FP32 compute
    return rocblas_launch<half,float,float>(dim_M, dim_N, dim_K, dev_wf_sz, FP16, FP32, FP32);
}

metrics rocblas_launch_bf16_bf16_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz)
{
    //BF16 multiply, BF16 accumulate, FP32 compute
    return rocblas_launch<rocblas_bfloat16,rocblas_bfloat16,float>(dim_M, dim_N, dim_K, dev_wf_sz, BF16, BF16, FP32);
}

metrics rocblas_launch_bf16_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz)
{
    //BF16 multiply, FP32 accumulate, FP32 compute
    return rocblas_launch<rocblas_bfloat16,float,float>(dim_M, dim_N, dim_K, dev_wf_sz, BF16, FP32, FP32);
}

metrics rocblas_launch_fp32_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz)
{
    //FP32 multiply, FP32 accumulate, FP32 compute
    return rocblas_launch<float,float,float>(dim_M, dim_N, dim_K, dev_wf_sz, FP32, FP32, FP32);
}

metrics rocblas_launch_tf32_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz)
{
    //TF32 multiply, FP32 accumulate, FP32 compute
    return rocblas_launch<rocblas_xfloat32,float,float>(dim_M, dim_N, dim_K, dev_wf_sz, TF32, FP32, FP32);
}

metrics rocblas_launch_fp64_fp64_fp64(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz)
{
    //FP64 multiply, FP64 accumulate, FP64 compute
    return rocblas_launch<double,double,double>(dim_M, dim_N, dim_K, dev_wf_sz, FP64, FP64, FP64);
}

metrics rocblas_launch_int8_int32_int32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz)
{
    //INT8 multiply, INT32 accumulate, INT32 compute
    return rocblas_launch<int8_t,int32_t,int32_t>(dim_M, dim_N, dim_K, dev_wf_sz, INT8, INT32, INT32);
}

template<typename TMul, typename TAcc, typename TScale>
inline metrics rocblas_launch(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, ptype mulOp, ptype accOp, ptype scaleOp)
{
    TMul   *dev_buf_A, *dev_buf_B; 
    TAcc   *dev_buf_C;
    TScale alpha;
    TScale beta;

    rocblas_handle handle;
    hipStream_t stream;

    rocblas_datatype mulType;
    rocblas_datatype accType;
    rocblas_datatype scaleType;

    float gpu_elapsed_time_ms;

    metrics run_metrics;

    // Creating rocBLAS handler
    hipErrchk(rocblas_create_handle(&handle));

    // Get rocBLAS stream
    hipErrchk(rocblas_get_stream(handle, &stream));

    // Setting-up alpha and beta
    alpha = (TScale) 1.00;
    beta  = (TScale) 0.00;
    hipErrchk(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    // Setting-up Precision
    if     (mulOp==FP16 && accOp==FP16 && scaleOp==FP16)    {mulType=rocblas_datatype_f16_r;  accType=rocblas_datatype_f16_r;  scaleType=rocblas_datatype_f16_r;}
    else if(mulOp==FP16 && accOp==FP16 && scaleOp==FP32)    {mulType=rocblas_datatype_f16_r;  accType=rocblas_datatype_f16_r;  scaleType=rocblas_datatype_f32_r;}
    else if(mulOp==FP16 && accOp==FP32 && scaleOp==FP32)    {mulType=rocblas_datatype_f16_r;  accType=rocblas_datatype_f32_r;  scaleType=rocblas_datatype_f32_r;}
    else if(mulOp==BF16 && accOp==FP32 && scaleOp==FP32)    {mulType=rocblas_datatype_bf16_r; accType=rocblas_datatype_f32_r;  scaleType=rocblas_datatype_f32_r;}
    else if(mulOp==BF16 && accOp==BF16 && scaleOp==FP32)    {mulType=rocblas_datatype_bf16_r; accType=rocblas_datatype_bf16_r; scaleType=rocblas_datatype_f32_r;}
    else if(mulOp==FP32 && accOp==FP32 && scaleOp==FP32)    {mulType=rocblas_datatype_f32_r;  accType=rocblas_datatype_f32_r;  scaleType=rocblas_datatype_f32_r;}
    else if(mulOp==TF32 && accOp==FP32 && scaleOp==FP32)    {mulType=rocblas_datatype_f32_r;  accType=rocblas_datatype_f32_r;  scaleType=rocblas_datatype_f32_r;}
    else if(mulOp==FP64 && accOp==FP64 && scaleOp==FP64)    {mulType=rocblas_datatype_f64_r;  accType=rocblas_datatype_f64_r;  scaleType=rocblas_datatype_f64_r;}
    else if(mulOp==INT8 && accOp==INT32 && scaleOp==INT32)  {mulType=rocblas_datatype_i8_r;   accType=rocblas_datatype_i32_r;  scaleType=rocblas_datatype_i32_r;}
    
    // Memory Allocation
    hipErrchk(hipMalloc((void**)&dev_buf_A, dim_M * dim_K * sizeof(TMul)));
    hipErrchk(hipMalloc((void**)&dev_buf_B, dim_K * dim_N * sizeof(TMul)));
    hipErrchk(hipMalloc((void**)&dev_buf_C, dim_M * dim_N * sizeof(TAcc)));

    // Launching rocBLAS GEMM
    gpu_elapsed_time_ms = 0.0;
    #pragma unroll
    for(int iter=0; iter < NUM_LOOPS_BLAS; iter++)
    {
        std::chrono::time_point start=std::chrono::high_resolution_clock::now();
        hipErrchk(rocblas_gemm_ex(handle,                       // rocBLAS handler
                                  rocblas_operation_none,       // Operation for MATRIX A (none, transpose, conjugate_transpose)
                                  rocblas_operation_none,       // Operation for MATRIX B (none, transpose, conjugate_transpose)
                                  dim_M,                        // dimension M
                                  dim_N,                        // dimension N
                                  dim_K,                        // dimension K
                                  &alpha,                       // scaling factor alpha where (alpha)x(AxB)
                                  dev_buf_A,                    // MATRIX A pointer on device memory
                                  mulType,                      // MATRIX A datatype
                                  dim_M,                        // MATRIX A leading dimension
                                  dev_buf_B,                    // MATRIX B pointer on device memory
                                  mulType,                      // MATRIX B datatype
                                  dim_K,                        // MATRIX B leading dimension
                                  &beta,                        // scaling factor beta where (beta)xC
                                  dev_buf_C,                    // MATRIX C pointer on device memory
                                  accType,                      // MATRIX C datatype
                                  dim_M,                        // MATRIX C leading dimension
                                  dev_buf_C,                    // MATRIX D pointer on device memory
                                  accType,                      // MATRIX D datatype
                                  dim_M,                        // MATRIX D leading dimension
                                  scaleType,                    // rocBLAS compute type
                                  rocblas_gemm_algo_standard,   // rocBLAS compute algorithm
                                  -1,                           // rocBLAS solution index
                                  rocblas_gemm_flags_none)      // rocBLAS flag
                );
        hipErrchk(hipStreamSynchronize(stream));
        std::chrono::time_point stop=std::chrono::high_resolution_clock::now();
        float elapsed_ms_host = (float) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000;
        gpu_elapsed_time_ms = gpu_elapsed_time_ms + elapsed_ms_host;
    }
    
    // Destroy rocBLAS handle
    hipErrchk(rocblas_destroy_handle(handle));

    hipErrchk(hipFree(dev_buf_A));
    hipErrchk(hipFree(dev_buf_B));
    hipErrchk(hipFree(dev_buf_C));

    // Storing the run metrics
    run_metrics.n_iter      = NUM_LOOPS_BLAS;
    run_metrics.n_thread    = 0;
    run_metrics.n_wg        = 0;
    run_metrics.n_wf        = 0;
    run_metrics.wf_size     = dev_wf_sz;
    run_metrics.wg_size     = 0;
    run_metrics.time_ms     = gpu_elapsed_time_ms;

    run_metrics.avg_clock   = 0;
    run_metrics.max_clock   = 0;
    run_metrics.min_clock   = 0;
    run_metrics.stdev_clock = 0;

    run_metrics.n_flops = (uint64_t) (2 * dim_M * dim_K * dim_N) * NUM_LOOPS_BLAS;
    run_metrics.n_bytes = (uint64_t) (2 * (dim_M*dim_K*sizeof(TMul) + dim_N*dim_K*sizeof(TMul) + dim_M*dim_N*sizeof(TAcc)))  * NUM_LOOPS_BLAS;

    run_metrics.gflop_per_s = (double) (run_metrics.n_flops / (run_metrics.time_ms*0.001))/1000000000;

    return run_metrics;
}


template<typename TMul, typename TAcc, typename TScale>
inline metrics rocblas_launch_float8_beta(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, ptype mulOp, ptype accOp, ptype scaleOp)
{
    TMul   *dev_buf_A, *dev_buf_B; 
    TAcc   *dev_buf_C;
    float  alpha;
    float  beta;

    rocblas_handle handle;
    hipStream_t stream;

    rocblas_datatype mulType;
    rocblas_datatype accType;
    rocblas_computetype scaleType;

    float gpu_elapsed_time_ms;

    metrics run_metrics;

    // Creating rocBLAS handler
    hipErrchk(rocblas_create_handle(&handle));

    // Get rocBLAS stream
    hipErrchk(rocblas_get_stream(handle, &stream));

    // Setting-up alpha and beta
    alpha = (float) 1.00;
    beta  = (float) 0.00;
    hipErrchk(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    // Setting-up Precision
    if     (mulOp==FP8  && accOp==FP32 && scaleOp==FP32)    {mulType=rocblas_datatype_f8_r;   accType=rocblas_datatype_f32_r;  scaleType=rocblas_compute_type_f8_f8_f32;}
    else if(mulOp==BF8  && accOp==FP32 && scaleOp==FP32)    {mulType=rocblas_datatype_bf8_r;  accType=rocblas_datatype_f32_r;  scaleType=rocblas_compute_type_bf8_bf8_f32;}

    // Memory Allocation
    hipErrchk(hipMalloc((void**)&dev_buf_A, dim_M * dim_K * sizeof(TMul)));
    hipErrchk(hipMalloc((void**)&dev_buf_B, dim_K * dim_N * sizeof(TMul)));
    hipErrchk(hipMalloc((void**)&dev_buf_C, dim_M * dim_N * sizeof(TAcc)));

    // Launching rocBLAS GEMM
    gpu_elapsed_time_ms = 0.0;
    #pragma unroll
    for(int iter=0; iter < NUM_LOOPS_BLAS; iter++)
    {
        std::chrono::time_point start=std::chrono::high_resolution_clock::now();
        hipErrchk(rocblas_gemm_ex3(handle,                      // rocBLAS handler
                                  rocblas_operation_none,       // Operation for MATRIX A (none, transpose, conjugate_transpose)
                                  rocblas_operation_none,       // Operation for MATRIX B (none, transpose, conjugate_transpose)
                                  dim_M,                        // dimension M
                                  dim_N,                        // dimension N
                                  dim_K,                        // dimension K
                                  &alpha,                       // scaling factor alpha where (alpha)x(AxB)
                                  dev_buf_A,                    // MATRIX A pointer on device memory
                                  mulType,                      // MATRIX A datatype
                                  dim_M,                        // MATRIX A leading dimension
                                  dev_buf_B,                    // MATRIX B pointer on device memory
                                  mulType,                      // MATRIX B datatype
                                  dim_K,                        // MATRIX B leading dimension
                                  &beta,                        // scaling factor beta where (beta)xC
                                  dev_buf_C,                    // MATRIX C pointer on device memory
                                  accType,                      // MATRIX C datatype
                                  dim_M,                        // MATRIX C leading dimension
                                  dev_buf_C,                    // MATRIX D pointer on device memory
                                  accType,                      // MATRIX D datatype
                                  dim_M,                        // MATRIX D leading dimension
                                  scaleType,                    // rocBLAS compute type
                                  rocblas_gemm_algo_standard,   // rocBLAS compute algorithm
                                  -1,                           // rocBLAS solution index
                                  rocblas_gemm_flags_none)      // rocBLAS flag
                );
        hipErrchk(hipStreamSynchronize(stream));
        std::chrono::time_point stop=std::chrono::high_resolution_clock::now();
        float elapsed_ms_host = (float) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000;
        gpu_elapsed_time_ms = gpu_elapsed_time_ms + elapsed_ms_host;
    }
    
    // Destroy rocBLAS handle
    hipErrchk(rocblas_destroy_handle(handle));

    hipErrchk(hipFree(dev_buf_A));
    hipErrchk(hipFree(dev_buf_B));
    hipErrchk(hipFree(dev_buf_C));

    // Storing the run metrics
    run_metrics.n_iter      = NUM_LOOPS_BLAS;
    run_metrics.n_thread    = 0;
    run_metrics.n_wg        = 0;
    run_metrics.n_wf        = 0;
    run_metrics.wf_size     = dev_wf_sz;
    run_metrics.wg_size     = 0;
    run_metrics.time_ms     = gpu_elapsed_time_ms;

    run_metrics.avg_clock   = 0;
    run_metrics.max_clock   = 0;
    run_metrics.min_clock   = 0;
    run_metrics.stdev_clock = 0;

    run_metrics.n_flops = (uint64_t) (2 * dim_M * dim_K * dim_N) * NUM_LOOPS_BLAS;
    run_metrics.n_bytes = (uint64_t) (2 * (dim_M*dim_K*sizeof(TMul) + dim_N*dim_K*sizeof(TMul) + dim_M*dim_N*sizeof(TAcc)))  * NUM_LOOPS_BLAS;

    run_metrics.gflop_per_s = (double) (run_metrics.n_flops / (run_metrics.time_ms*0.001))/1000000000;

    return run_metrics;
}