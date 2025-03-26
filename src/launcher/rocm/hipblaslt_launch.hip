#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt_xfloat32.h>
#include <GPU_Roofline_Tools/launcher/rocm/hipblaslt_launch.hip.h>
#include <GPU_Roofline_Tools/utils/rocm/hipcheck.hip.h>
#include <GPU_Roofline_Tools/utils/common/ptype.h>
#include <GPU_Roofline_Tools/utils/common/metrics.h>
#include <GPU_Roofline_Tools/utils/common/global.h>


metrics hipblaslt_launch_fp8_fp8_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace)
{
    //FP8 multiply, FP8 accumulate, FP32 compute
    return hipblaslt_launch<hipblaslt_f8_fnuz,hipblaslt_f8_fnuz,float>(dim_M, dim_N, dim_K, dev_wf_sz, FP8, FP8, FP32, use_workspace);
}

metrics hipblaslt_launch_fp8_fp16_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace)
{
    //FP8 multiply, FP16 accumulate, FP32 compute
    return hipblaslt_launch<hipblaslt_f8_fnuz,hipblasLtHalf,float>(dim_M, dim_N, dim_K, dev_wf_sz, FP8, FP16, FP32, use_workspace);
}

metrics hipblaslt_launch_fp8_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace)
{
    //FP8 multiply, FP32 accumulate, FP32 compute
    return hipblaslt_launch<hipblaslt_f8_fnuz,float,float>(dim_M, dim_N, dim_K, dev_wf_sz, FP8, FP32, FP32, use_workspace);
}

metrics hipblaslt_launch_bf8_bf8_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace)
{
    //BF8 multiply, BF8 accumulate, FP32 compute
    return hipblaslt_launch<hipblaslt_bf8_fnuz,hipblaslt_bf8_fnuz,float>(dim_M, dim_N, dim_K, dev_wf_sz, BF8, BF8, FP32, use_workspace);
}

metrics hipblaslt_launch_bf8_fp16_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace)
{
    //BF8 multiply, FP16 accumulate, FP32 compute
    return hipblaslt_launch<hipblaslt_bf8_fnuz,hipblasLtHalf,float>(dim_M, dim_N, dim_K, dev_wf_sz, BF8, FP16, FP32, use_workspace);
}

metrics hipblaslt_launch_bf8_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace)
{
    //BF8 multiply, FP32 accumulate, FP32 compute
    return hipblaslt_launch<hipblaslt_bf8_fnuz,float,float>(dim_M, dim_N, dim_K, dev_wf_sz, BF8, FP32, FP32, use_workspace);
}

metrics hipblaslt_launch_fp16_fp16_fp16(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace)
{
    //FP16 multiply, FP16 accumulate, FP16 compute
    return hipblaslt_launch<hipblasLtHalf,hipblasLtHalf,_Float16>(dim_M, dim_N, dim_K, dev_wf_sz, FP16, FP16, FP16, use_workspace);
}

metrics hipblaslt_launch_bf16_bf16_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace)
{
    //BF16 multiply, BF16 accumulate, FP32 compute
    return hipblaslt_launch<hipblasLtBfloat16,hipblasLtBfloat16,float>(dim_M, dim_N, dim_K, dev_wf_sz, BF16, BF16, FP32, use_workspace);
}

metrics hipblaslt_launch_bf16_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace)
{
    //BF16 multiply, FP32 accumulate, FP32 compute
    return hipblaslt_launch<hipblasLtBfloat16,float,float>(dim_M, dim_N, dim_K, dev_wf_sz, BF16, FP32, FP32, use_workspace);
}

metrics hipblaslt_launch_fp16_fp16_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace)
{
    //FP16 multiply, FP16 accumulate, FP32 compute
    return hipblaslt_launch<hipblasLtHalf,hipblasLtHalf,float>(dim_M, dim_N, dim_K, dev_wf_sz, FP16, FP16, FP32, use_workspace);
}

metrics hipblaslt_launch_fp16_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace)
{
    //FP16 multiply, FP32 accumulate, FP32 compute
    return hipblaslt_launch<hipblasLtHalf,float,float>(dim_M, dim_N, dim_K, dev_wf_sz, FP16, FP32, FP32, use_workspace);
}

metrics hipblaslt_launch_tf32_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace)
{
    //TF32 multiply, FP32 accumulate, FP32 compute
    return hipblaslt_launch<hipblasLtXfloat32,float,float>(dim_M, dim_N, dim_K, dev_wf_sz, TF32, FP32, FP32, use_workspace);
}

metrics hipblaslt_launch_fp32_fp32_fp32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace)
{
    //FP32 multiply, FP32 accumulate, FP32 compute
    return hipblaslt_launch<float,float,float>(dim_M, dim_N, dim_K, dev_wf_sz, FP32, FP32, FP32, use_workspace);
}

metrics hipblaslt_launch_fp64_fp64_fp64(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace)
{
    //FP64 multiply, FP64 accumulate, FP64 compute
    return hipblaslt_launch<double,double,double>(dim_M, dim_N, dim_K, dev_wf_sz, FP64, FP64, FP64, use_workspace);
}

metrics hipblaslt_launch_int8_int32_int32(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, bool use_workspace)
{
    //INT8 multiply, INT32 accumulate, INT32 compute
    return hipblaslt_launch<int8_t,int32_t,int32_t>(dim_M, dim_N, dim_K, dev_wf_sz, INT8, INT32, INT32, use_workspace);
}
/*

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
*/

template<typename TMul, typename TAcc, typename TScale>
inline metrics hipblaslt_launch(uint64_t dim_M, uint64_t dim_N, uint64_t dim_K, uint32_t dev_wf_sz, ptype mulOp, ptype accOp, ptype scaleOp, bool use_workspace)
{
    TMul   *dev_buf_A, *dev_buf_B; 
    TAcc   *dev_buf_C;
    TScale alpha;
    TScale beta;

    hipblasLtHandle_t handle;
    hipStream_t stream;

    hipDataType  mulType;
    hipDataType  accType;
    hipDataType  scaleType;

    hipblasComputeType_t computeType;

    float gpu_elapsed_time_ms;

    metrics run_metrics;

    // Create hip stream
    hipErrchk(hipStreamCreate(&stream));

    // Creating hipblaslt handler
    hipErrchk(hipblasLtCreate(&handle));

    // Setting-up alpha and beta
    alpha = (TScale) 1.00;
    beta  = (TScale) 0.00;

    // Setting-up Precision
    if     (mulOp==FP16 && accOp==FP16 && scaleOp==FP16)    {mulType=HIP_R_16F;           accType=HIP_R_16F;          scaleType=HIP_R_16F; computeType=HIPBLAS_COMPUTE_16F;}
    else if(mulOp==FP16 && accOp==FP16 && scaleOp==FP32)    {mulType=HIP_R_16F;           accType=HIP_R_16F;          scaleType=HIP_R_32F; computeType=HIPBLAS_COMPUTE_32F;}
    else if(mulOp==FP16 && accOp==FP32 && scaleOp==FP32)    {mulType=HIP_R_16F;           accType=HIP_R_32F;          scaleType=HIP_R_32F; computeType=HIPBLAS_COMPUTE_32F;}
    else if(mulOp==BF16 && accOp==FP32 && scaleOp==FP32)    {mulType=HIP_R_16BF;          accType=HIP_R_32F;          scaleType=HIP_R_32F; computeType=HIPBLAS_COMPUTE_32F;}
    else if(mulOp==BF16 && accOp==BF16 && scaleOp==FP32)    {mulType=HIP_R_16BF;          accType=HIP_R_16BF;         scaleType=HIP_R_32F; computeType=HIPBLAS_COMPUTE_32F;}
    else if(mulOp==FP32 && accOp==FP32 && scaleOp==FP32)    {mulType=HIP_R_32F;           accType=HIP_R_32F;          scaleType=HIP_R_32F; computeType=HIPBLAS_COMPUTE_32F;}
    else if(mulOp==FP8  && accOp==FP8  && scaleOp==FP32)    {mulType=HIP_R_8F_E4M3_FNUZ;  accType=HIP_R_8F_E4M3_FNUZ; scaleType=HIP_R_32F; computeType=HIPBLAS_COMPUTE_32F;}
    else if(mulOp==FP8  && accOp==FP16 && scaleOp==FP32)    {mulType=HIP_R_8F_E4M3_FNUZ;  accType=HIP_R_16F;          scaleType=HIP_R_32F; computeType=HIPBLAS_COMPUTE_32F;}
    else if(mulOp==FP8  && accOp==FP32 && scaleOp==FP32)    {mulType=HIP_R_8F_E4M3_FNUZ;  accType=HIP_R_32F;          scaleType=HIP_R_32F; computeType=HIPBLAS_COMPUTE_32F;}
    else if(mulOp==BF8  && accOp==BF8 && scaleOp==FP32)     {mulType=HIP_R_8F_E5M2_FNUZ;  accType=HIP_R_8F_E5M2_FNUZ; scaleType=HIP_R_32F; computeType=HIPBLAS_COMPUTE_32F;}
    else if(mulOp==BF8  && accOp==FP16 && scaleOp==FP32)    {mulType=HIP_R_8F_E5M2_FNUZ;  accType=HIP_R_16F;          scaleType=HIP_R_32F; computeType=HIPBLAS_COMPUTE_32F;}
    else if(mulOp==BF8  && accOp==FP32 && scaleOp==FP32)    {mulType=HIP_R_8F_E5M2_FNUZ;  accType=HIP_R_32F;          scaleType=HIP_R_32F; computeType=HIPBLAS_COMPUTE_32F;}
    else if(mulOp==TF32 && accOp==FP32 && scaleOp==FP32)    {mulType=HIP_R_32F;           accType=HIP_R_32F;          scaleType=HIP_R_32F; computeType=HIPBLAS_COMPUTE_32F_FAST_TF32;}
    else if(mulOp==FP64 && accOp==FP64 && scaleOp==FP64)    {mulType=HIP_R_64F;           accType=HIP_R_64F;          scaleType=HIP_R_64F; computeType=HIPBLAS_COMPUTE_32F;}
    else if(mulOp==INT8 && accOp==INT32 && scaleOp==INT32)  {mulType=HIP_R_8I;            accType=HIP_R_32I;          scaleType=HIP_R_32I; computeType=HIPBLAS_COMPUTE_32I;}
    
    // Memory Allocation
    hipErrchk(hipMalloc((void**)&dev_buf_A, dim_M * dim_K * sizeof(TMul)));
    hipErrchk(hipMalloc((void**)&dev_buf_B, dim_K * dim_N * sizeof(TMul)));
    hipErrchk(hipMalloc((void**)&dev_buf_C, dim_M * dim_N * sizeof(TAcc)));

    // HipblasLT layout
    hipblasLtMatrixLayout_t matA, matB, matC;
    hipErrchk(hipblasLtMatrixLayoutCreate(&matA, mulType, dim_M, dim_K, dim_M));
    hipErrchk(hipblasLtMatrixLayoutCreate(&matB, mulType, dim_K, dim_N, dim_K));
    hipErrchk(hipblasLtMatrixLayoutCreate(&matC, accType, dim_M, dim_N, dim_M));

    // HipblasLT matmul description
    hipblasLtMatmulDesc_t matmuldesc;
    hipErrchk(hipblasLtMatmulDescCreate(&matmuldesc, computeType, scaleType));

    hipblasOperation_t operations = HIPBLAS_OP_N;
    hipErrchk(hipblasLtMatmulDescSetAttribute(matmuldesc, HIPBLASLT_MATMUL_DESC_TRANSA, &operations, sizeof(operations)));
    hipErrchk(hipblasLtMatmulDescSetAttribute(matmuldesc, HIPBLASLT_MATMUL_DESC_TRANSB, &operations, sizeof(operations)));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    hipErrchk(hipblasLtMatmulDescSetAttribute(matmuldesc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    hipblasLtPointerMode_t pointer = HIPBLASLT_POINTER_MODE_HOST;
    hipErrchk(hipblasLtMatmulDescSetAttribute(matmuldesc, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pointer, sizeof(pointer)));
    

    // Matmul Preference and Workspace Size
    hipblasLtMatmulPreference_t pref;

    uint64_t max_workspace_size;
    if(use_workspace) {max_workspace_size = 128 * 1024 * 1024;} //The default workspace size is 32 MiB; MI300 and newer defaults to 128 MiB.
    else              {max_workspace_size = 0;}

    hipErrchk(hipblasLtMatmulPreferenceCreate(&pref));
    hipErrchk(hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));

    // Finding Algorithm
    const int request_solutions = 1;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int returnedAlgoCount = 0;
    hipErrchk(hipblasLtMatmulAlgoGetHeuristic(handle,     // hipBLASLT handler
                                              matmuldesc, // hipBLASLT Matmul Description
                                              matA,       // MATRIX A layout and data type
                                              matB,       // MATRIX B layout and data type
                                              matC,       // MATRIX C layout and data type
                                              matC,       // MATRIX C layout and data type
                                              pref,       // Matmul Preference, passing the workspace size
                                              request_solutions, // Number of solution requested
                                              heuristicResult,   // Algorithm returned
                                              &returnedAlgoCount)); // Number of algorithm returned

    //if(returnedAlgoCount == 0){std::cerr <<"[ERR!] No valid solution found for GEMM using hipBLASLT!" << std::endl; exit(1);}

    uint64_t workspace_size;
    void* dev_workspace;
    
    if(use_workspace)
    {
        for(int i = 0; i < returnedAlgoCount; i++) {workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);}
        hipErrchk(hipMalloc((void**)&dev_workspace, workspace_size));
    }
    else
    {
        workspace_size=0;
    }

    // Launching hipblaslt GEMM
    gpu_elapsed_time_ms = 0.0;
    #pragma unroll
    for(int iter=0; iter < NUM_LOOPS_BLAS; iter++)
    {
        std::chrono::time_point start=std::chrono::high_resolution_clock::now();

        hipErrchk(hipblasLtMatmul(handle,     // hipBLASLT handler
                                  matmuldesc, // hipBLASLT Matmul Description
                                  &alpha,     // scaling factor alpha where (alpha)x(AxB)
                                  dev_buf_A,  // MATRIX A pointer on device memory
                                  matA,       // MATRIX A layout and data type
                                  dev_buf_B,  // MATRIX B pointer on device memory
                                  matB,       // MATRIX B layout and data type
                                  &beta,      // scaling factor beta where (beta)xC
                                  dev_buf_C,  // MATRIX C pointer on device memory
                                  matC,       // MATRIX C layout and data type
                                  dev_buf_C,  // MATRIX D pointer on device memory
                                  matC,       // MATRIX D layout and data type
                                  &heuristicResult[0].algo, // Algorithm result from heuristic.
                                  dev_workspace,  // Allocated memory for workspace
                                  workspace_size, // Workspace size based on chosen algo.
                                  stream)     // hipStream
                );

        hipErrchk(hipStreamSynchronize(stream));
        std::chrono::time_point stop=std::chrono::high_resolution_clock::now();
        float elapsed_ms_host = (float) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000;
        gpu_elapsed_time_ms = gpu_elapsed_time_ms + elapsed_ms_host;
    }

    hipErrchk(hipblasLtMatrixLayoutDestroy(matA));
    hipErrchk(hipblasLtMatrixLayoutDestroy(matB));
    hipErrchk(hipblasLtMatrixLayoutDestroy(matC));
    hipErrchk(hipblasLtMatmulDescDestroy(matmuldesc));
    hipErrchk(hipblasLtMatmulPreferenceDestroy(pref));
    
    // Destroy hipblasLT handle
    hipErrchk(hipblasLtDestroy(handle));

    hipErrchk(hipFree(dev_buf_A));
    hipErrchk(hipFree(dev_buf_B));
    hipErrchk(hipFree(dev_buf_C));

    if(use_workspace) {hipErrchk(hipFree(dev_workspace));}

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