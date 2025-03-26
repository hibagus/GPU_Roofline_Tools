#include <argparse/argparse.hpp>
#include <ctime>
#include <GPU_Roofline_Tools/launcher/rocm/kernel_launch_vector.hip.h>
#include <GPU_Roofline_Tools/launcher/rocm/kernel_device_init.hip.h>
#include <GPU_Roofline_Tools/launcher/rocm/kernel_launch_wmma.hip.h>
#include <GPU_Roofline_Tools/launcher/rocm/rocblas_launch.hip.h>
#include <GPU_Roofline_Tools/launcher/rocm/hipblaslt_launch.hip.h>
#include <GPU_Roofline_Tools/utils/common/optype.h>
#include <GPU_Roofline_Tools/utils/common/ptype.h>
#include <GPU_Roofline_Tools/utils/common/metrics.h>
#include <GPU_Roofline_Tools/utils/common/blaslib.h>

int main(int argc, char *argv[])
{
    // Program Title
    std::cout << "[INFO] GPU Roofline Tools \n";
    std::cout << "[INFO] Version 1.0.0 (C)2024 Bagus Hanindhito \n";
    std::cout << "[INFO] Built for AMD Instinct MI300X (gfx942) GPUs\n";

    initHIPDevice();
    
    // Arguments Parser
    argparse::ArgumentParser program(argv[0], "1.0.0", argparse::default_arguments::help);
    
    // ================== Execution control ================== 
    // For all type execution
    program.add_argument("--device")
            .help("Select device (GPU) where the benchmark is run according to the index number provided by rocm-smi.")
            .scan<'i', int>()
            .default_value(0)
            .metavar("DEVICE");
    program.add_argument("--operations")
            .help("Select operation types: V_ADD1, V_ADD2, V_MUL1, V_MUL2, V_FMA3, V_FMA2, V_FMA1, M_WMMA, M_BLAS")
            .default_value(std::string("V_FMA3"))
            .metavar("ops");
    program.add_argument("--blas_lib")
            .help("Select library for BLAS: ROCBLAS, HIPBLASLT")
            .default_value(std::string("ROCBLAS"))
            .metavar("blas_lib");

    // Vector Execution
    program.add_argument("--vector-data-type")
            .help("Select vector data type: fp64, fp32, fp16, or bf16. This is used for vector operations on Stream Processors (SP).")
            .default_value(std::string("fp16"))
            .metavar("v_dtype");

    // Matrix Execution
    program.add_argument("--matrix-mult-type")
            .help("Select matrix multiplication data type: fp64, fp32, tf32, fp16, bf16, fp8, bf8, int8, or int32. This is used for matrix operations on Matrix Cores (MC).")
            .default_value(std::string("fp16"))
            .metavar("m_multype");

    program.add_argument("--matrix-accum-type")
            .help("Select matrix accumulation data type: fp64, fp32, fp16, or int32. This is used for matrix operations on Matrix Cores (MC).")
            .default_value(std::string("fp32"))
            .metavar("m_acctype");

    program.add_argument("--matrix-scale-type")
            .help("Select matrix accumulation data type: fp64, fp32, fp16, or int32. This is used for matrix operations on Matrix Cores (MC).")
            .default_value(std::string("fp32"))
            .metavar("m_scaletype");

    // rocBLAS/hipBLASLt Execution
    program.add_argument("--dim-M")
            .help("M dimension of GEMM for rocBLAS and hipBLASLt.")
            .scan<'i', uint64_t>()
            .default_value(1024UL)
            .metavar("M_DIM");

   program.add_argument("--dim-K")
            .help("K dimension of GEMM for rocBLAS and hipBLASLt.")
            .scan<'i', uint64_t>()
            .default_value(1024UL)
            .metavar("K_DIM");

   program.add_argument("--dim-N")
            .help("N dimension of GEMM for rocBLAS and hipBLASLt.")
            .scan<'i', uint64_t>()
            .default_value(1024UL)
            .metavar("N_DIM");
    
   program.add_argument("--use-workspace")
            .help("Allocate workspace in device memory for hipBLASLt: true or false.")
            .default_value(std::string("false"))
            .metavar("use_workspace");
    

    // ================== Wavefront control ================== 
    program.add_argument("--min-wavefront")
            .help("Minimum number of wavefront (warp) per workgroup (block). Must be positive integer smaller than or equal to MAX_WF.")
            .scan<'i', int>()
            .default_value(1)
            .metavar("MIN_WF");
    program.add_argument("--max-wavefront")
            .help("Maximum number of wavefront (warp) per workgroup (block). Must be positive integer larger than or equal to MIN_WF.")
            .scan<'i', int>()
            .default_value(1)
            .metavar("MAX_WF");
    program.add_argument("--step-wavefront")
            .help("Increment step for the number of wavefront (warp) per workgroup (block) between MIN_WF and MAX_WF for sweeping.")
            .scan<'i', int>()
            .default_value(1)
            .metavar("STEP_WF");
    
    // ==================  Workgroup Control ================== 
    program.add_argument("--min-workgroup")
            .help("Minimum number of workgroup (block) per NDRange (grid). Must be positive integer smaller than or equal to MAX_WG.")
            .scan<'i', int>()
            .default_value(1)
            .metavar("MIN_WG");
    program.add_argument("--max-workgroup")
            .help("Maximum number of workgroup (block) per NDRange (grid). Must be positive integer larger than or equal to MIN_WG.")
            .scan<'i', int>()
            .default_value(1)
            .metavar("MAX_WG");
    program.add_argument("--step-workgroup")
            .help("Increment step for the number of workgroup (block) per NDRange (grid) between MIN_WG and MAX_WG for sweeping.")
            .scan<'i', int>()
            .default_value(1)
            .metavar("STEP_WG");
    
    // ================== Parsing the arguments ================== 
    try 
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) 
    {
        std::cerr << "[ERR!] Argument parsing error: " << err.what() << std::endl << std::endl << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    // Device Selection
    int device = program.get<int>("--device");

    // Operation Type
    std::string str_operations  = program.get<std::string>("--operations");

    // Data Type
    std::string str_v_dtype     = program.get<std::string>("--vector-data-type");
    std::string str_m_multype   = program.get<std::string>("--matrix-mult-type");
    std::string str_m_acctype   = program.get<std::string>("--matrix-accum-type");
    std::string str_m_scaletype = program.get<std::string>("--matrix-scale-type");

    // rocBLAS GEMM Dimension
    uint64_t dim_M = program.get<uint64_t>("--dim-M");
    uint64_t dim_K = program.get<uint64_t>("--dim-K");
    uint64_t dim_N = program.get<uint64_t>("--dim-N");
    
    // Workgroup Control
    int min_workgroup  = program.get<int>("--min-workgroup");
    int max_workgroup  = program.get<int>("--max-workgroup");
    int step_workgroup = program.get<int>("--step-workgroup");

    // Wavefront Control
    int min_wavefront  = program.get<int>("--min-wavefront");
    int max_wavefront  = program.get<int>("--max-wavefront");
    int step_wavefront = program.get<int>("--step-wavefront");

    // BLAS Library
    std::string str_blas_lib      = program.get<std::string>("--blas_lib"); 
    std::string str_use_workspace = program.get<std::string>("--use-workspace"); 

    // Argument Validation
    // Device Selection
    setHIPDevice(device);
    uint32_t dev_max_wg_sz = getMaxWorkgroupSize(device);
    uint32_t dev_wf_sz     = getWaveFrontSize(device);

    // Numerical Validation
    if (min_workgroup<1) {std::cerr << "[ERR!] MIN_WG must be positive number!" << std::endl; exit(1);}
    if (max_workgroup<1) {std::cerr << "[ERR!] MAX_WG must be positive number!" << std::endl; exit(1);}
    if (step_workgroup<1){std::cerr << "[ERR!] STEP_WG must be positive number!" << std::endl; exit(1);}

    if (min_wavefront<1) {std::cerr << "[ERR!] MIN_WF must be positive number!" << std::endl; exit(1);}
    if (max_wavefront<1) {std::cerr << "[ERR!] MAX_WF must be positive number!" << std::endl; exit(1);}
    if (step_wavefront<1){std::cerr << "[ERR!] STEP_WF must be positive number!" << std::endl; exit(1);}

    ptype v_dtype;
    ptype m_multype;
    ptype m_acctype;
    ptype m_scaletype;

    // Data Type
    if       (str_v_dtype=="fp64"){v_dtype=FP64;}
    else if  (str_v_dtype=="fp32"){v_dtype=FP32;}
    else if  (str_v_dtype=="fp16"){v_dtype=FP16;}
    else if  (str_v_dtype=="bf16"){v_dtype=BF16;}
    else     {std::cerr <<"[ERR!] Argument parsing error: Unsupported vector data type!" << std::endl; exit(1);}

    if       (str_m_multype=="fp64") {m_multype=FP64;}
    else if  (str_m_multype=="fp32") {m_multype=FP32;}
    else if  (str_m_multype=="tf32") {m_multype=TF32;}
    else if  (str_m_multype=="fp16") {m_multype=FP16;}
    else if  (str_m_multype=="fp8")  {m_multype=FP8;}
    else if  (str_m_multype=="bf16") {m_multype=BF16;}
    else if  (str_m_multype=="bf8")  {m_multype=BF8;}
    else if  (str_m_multype=="int8") {m_multype=INT8;}
    else if  (str_m_multype=="int32"){m_multype=INT32;}
    else     {std::cerr <<"[ERR!] Argument parsing error: Unsupported multiplication matrix data type!" << std::endl; exit(1);}

    if       (str_m_acctype=="fp64") {m_acctype=FP64;}
    else if  (str_m_acctype=="fp32") {m_acctype=FP32;}
    else if  (str_m_acctype=="fp16") {m_acctype=FP16;}
    else if  (str_m_acctype=="bf16") {m_acctype=BF16;}
    else if  (str_m_acctype=="fp8")  {m_acctype=FP8;}
    else if  (str_m_acctype=="bf8")  {m_acctype=BF8;}
    else if  (str_m_acctype=="int32"){m_acctype=INT32;}
    else     {std::cerr <<"[ERR!] Argument parsing error: Unsupported accumulation matrix data type!" << std::endl; exit(1);}

    if       (str_m_scaletype=="fp64") {m_scaletype=FP64;}
    else if  (str_m_scaletype=="fp32") {m_scaletype=FP32;}
    else if  (str_m_scaletype=="fp16") {m_scaletype=FP16;}
    else if  (str_m_scaletype=="int32"){m_scaletype=INT32;}
    else     {std::cerr <<"[ERR!] Argument parsing error: Unsupported scaling matrix data type!" << std::endl; exit(1);}

    // Operations
    optype operations;
    if       (str_operations=="V_ADD1"){operations=V_ADD1;}
    //else if  (str_operations=="V_ADD1"){operations=V_ADD1;}
    else if  (str_operations=="V_ADD2"){operations=V_ADD2;}
    else if  (str_operations=="V_MUL1"){operations=V_MUL1;}
    else if  (str_operations=="V_MUL2"){operations=V_MUL2;}
    else if  (str_operations=="V_FMA3"){operations=V_FMA3;}
    else if  (str_operations=="V_FMA2"){operations=V_FMA2;}
    else if  (str_operations=="V_FMA1"){operations=V_FMA1;}
    else if  (str_operations=="M_WMMA"){operations=M_WMMA;}
    else if  (str_operations=="M_BLAS"){operations=M_BLAS;}
    else     {std::cerr <<"[ERR!] Argument parsing error: Unsupported operation!" << std::endl; exit(1);}

    // BLAS Library
    bool use_workspace;
    blaslib lib;

    if       (str_blas_lib=="ROCBLAS"){lib=ROCBLAS;}
    else if  (str_blas_lib=="HIPBLASLT"){lib=HIPBLASLT;}
    else     {std::cerr <<"[ERR!] Argument parsing error: Unsupported BLAS Library!" << std::endl; exit(1);}

    if       (str_use_workspace=="true") {use_workspace=true;}
    else if  (str_use_workspace=="false"){use_workspace=false;}
    else     {std::cerr <<"[ERR!] Argument parsing error: Unsupported option for hipBLASLt workspace!" << std::endl; exit(1);}

    // For storing the run metrics
    std::vector<metrics> all_metrics;

    // Workgroup Sweep
    if(operations==M_BLAS) // with rocBLAS we cannot control wavefront/workgroup
    {
        metrics run_metrics;
        if(lib==ROCBLAS)
        {
            std::cout << "[INFO] (" << std::time(nullptr) << ") Running rocBLAS for GEMM " << dim_M << "x" << dim_N << "x" << dim_K << " size and " << str_m_multype << "/" << str_m_acctype << "/" << str_m_scaletype << " precisions" << std::endl; 
            if     (m_multype==FP64 && m_acctype==FP64   && m_scaletype==FP64)   {run_metrics=rocblas_launch_fp64_fp64_fp64(dim_M, dim_N, dim_K, dev_wf_sz);}
            else if(m_multype==FP32 && m_acctype==FP32   && m_scaletype==FP32)   {run_metrics=rocblas_launch_fp32_fp32_fp32(dim_M, dim_N, dim_K, dev_wf_sz);}
            else if(m_multype==TF32 && m_acctype==FP32   && m_scaletype==FP32)   {run_metrics=rocblas_launch_tf32_fp32_fp32(dim_M, dim_N, dim_K, dev_wf_sz);}
            else if(m_multype==FP16 && m_acctype==FP32   && m_scaletype==FP32)   {run_metrics=rocblas_launch_fp16_fp32_fp32(dim_M, dim_N, dim_K, dev_wf_sz);}
            else if(m_multype==FP16 && m_acctype==FP16   && m_scaletype==FP32)   {run_metrics=rocblas_launch_fp16_fp16_fp32(dim_M, dim_N, dim_K, dev_wf_sz);}
            else if(m_multype==FP16 && m_acctype==FP16   && m_scaletype==FP16)   {run_metrics=rocblas_launch_fp16_fp16_fp16(dim_M, dim_N, dim_K, dev_wf_sz);}
            else if(m_multype==BF16 && m_acctype==FP32   && m_scaletype==FP32)   {run_metrics=rocblas_launch_bf16_fp32_fp32(dim_M, dim_N, dim_K, dev_wf_sz);}
            else if(m_multype==BF16 && m_acctype==BF16   && m_scaletype==FP32)   {run_metrics=rocblas_launch_bf16_bf16_fp32(dim_M, dim_N, dim_K, dev_wf_sz);}
            //else if(m_multype==FP8  && m_acctype==FP32   && m_scaletype==FP32)   {run_metrics=rocblas_launch_fp8_fp32_fp32(dim_M, dim_N, dim_K, dev_wf_sz);}
            //else if(m_multype==BF8  && m_acctype==FP32   && m_scaletype==FP32)   {run_metrics=rocblas_launch_bf8_fp32_fp32(dim_M, dim_N, dim_K, dev_wf_sz);}
            else if(m_multype==INT8 && m_acctype==INT32  && m_scaletype==INT32)  {run_metrics=rocblas_launch_int8_int32_int32(dim_M, dim_N, dim_K, dev_wf_sz);}
            else {std::cerr <<"[ERR!] Unsupported multiply/accumulation data type combinations!" << std::endl; exit(1);}
        }
        else if (lib==HIPBLASLT)
        {
             std::cout << "[INFO] (" << std::time(nullptr) << ") Running hipBLASLt for GEMM " << dim_M << "x" << dim_N << "x" << dim_K << " size and " << str_m_multype << "/" << str_m_acctype << "/" << str_m_scaletype << " precisions" << std::endl; 
             if     (m_multype==FP64 && m_acctype==FP64   && m_scaletype==FP64)   {run_metrics=hipblaslt_launch_fp64_fp64_fp64(dim_M, dim_N, dim_K, dev_wf_sz, use_workspace);}
             else if(m_multype==FP32 && m_acctype==FP32   && m_scaletype==FP32)   {run_metrics=hipblaslt_launch_fp32_fp32_fp32(dim_M, dim_N, dim_K, dev_wf_sz, use_workspace);}
             else if(m_multype==TF32 && m_acctype==FP32   && m_scaletype==FP32)   {run_metrics=hipblaslt_launch_tf32_fp32_fp32(dim_M, dim_N, dim_K, dev_wf_sz, use_workspace);}
             else if(m_multype==BF16 && m_acctype==FP32   && m_scaletype==FP32)   {run_metrics=hipblaslt_launch_bf16_fp32_fp32(dim_M, dim_N, dim_K, dev_wf_sz, use_workspace);}
             else if(m_multype==BF16 && m_acctype==BF16   && m_scaletype==FP32)   {run_metrics=hipblaslt_launch_bf16_bf16_fp32(dim_M, dim_N, dim_K, dev_wf_sz, use_workspace);}
             else if(m_multype==FP16 && m_acctype==FP32   && m_scaletype==FP32)   {run_metrics=hipblaslt_launch_fp16_fp32_fp32(dim_M, dim_N, dim_K, dev_wf_sz, use_workspace);}
             else if(m_multype==FP16 && m_acctype==FP16   && m_scaletype==FP32)   {run_metrics=hipblaslt_launch_fp16_fp16_fp32(dim_M, dim_N, dim_K, dev_wf_sz, use_workspace);}
             else if(m_multype==FP16 && m_acctype==FP16   && m_scaletype==FP16)   {run_metrics=hipblaslt_launch_fp16_fp16_fp16(dim_M, dim_N, dim_K, dev_wf_sz, use_workspace);}
             else if(m_multype==BF8 && m_acctype==BF8     && m_scaletype==FP32)   {run_metrics=hipblaslt_launch_bf8_bf8_fp32(dim_M, dim_N, dim_K, dev_wf_sz, use_workspace);}
             else if(m_multype==BF8 && m_acctype==FP16    && m_scaletype==FP32)   {run_metrics=hipblaslt_launch_bf8_fp16_fp32(dim_M, dim_N, dim_K, dev_wf_sz, use_workspace);}
             else if(m_multype==BF8 && m_acctype==FP32    && m_scaletype==FP32)   {run_metrics=hipblaslt_launch_bf8_fp32_fp32(dim_M, dim_N, dim_K, dev_wf_sz, use_workspace);}
             else if(m_multype==FP8 && m_acctype==FP8     && m_scaletype==FP32)   {run_metrics=hipblaslt_launch_fp8_fp8_fp32(dim_M, dim_N, dim_K, dev_wf_sz, use_workspace);}
             else if(m_multype==FP8 && m_acctype==FP16    && m_scaletype==FP32)   {run_metrics=hipblaslt_launch_fp8_fp16_fp32(dim_M, dim_N, dim_K, dev_wf_sz, use_workspace);}
             else if(m_multype==FP8 && m_acctype==FP32    && m_scaletype==FP32)   {run_metrics=hipblaslt_launch_fp8_fp32_fp32(dim_M, dim_N, dim_K, dev_wf_sz, use_workspace);}
             else if(m_multype==INT8 && m_acctype==INT32  && m_scaletype==INT32)  {run_metrics=hipblaslt_launch_int8_int32_int32(dim_M, dim_N, dim_K, dev_wf_sz, use_workspace);}
             else {std::cerr <<"[ERR!] Unsupported multiply/accumulation data type combinations!" << std::endl; exit(1);}
        }

        all_metrics.push_back(run_metrics);
    }
    else //Vector Ops and M_WMMA
    {
        for (int num_wg=min_workgroup; num_wg<=max_workgroup; num_wg=num_wg+step_workgroup)
        {
            // Wavefront Sweep
            for(int num_wf=min_wavefront; num_wf<=max_wavefront; num_wf=num_wf+step_wavefront)
            {
               std::cout << "[INFO] (" << std::time(nullptr) << ") Running with " << num_wf << " wavefront(s) per workgroup and " << num_wg << " workgroup(s)" << std::endl; 
               int wg_sz = num_wf * dev_wf_sz;
               if (wg_sz > dev_max_wg_sz)
               {
                    // Invalid run
                    std::cerr << "[ERR!]: Number of wavefront per workgroup (" << num_wf << ") results in total thread per workgroup exceeding the maximum value. Run will be skipped.\n";
               }
               else
               {
                    metrics run_metrics;
                    // Kernel launch
                    if (operations==V_ADD1 || operations==V_ADD2 || operations==V_MUL1 || operations==V_MUL2 || operations==V_FMA3 || operations==V_FMA2 || operations==V_FMA1)
                    {
                         if     (v_dtype==FP64){run_metrics=kernel_launch_vector_fp64(num_wf, num_wg, dev_wf_sz, operations);}
                         else if(v_dtype==FP32){run_metrics=kernel_launch_vector_fp32(num_wf, num_wg, dev_wf_sz, operations);}
                         else if(v_dtype==FP16){run_metrics=kernel_launch_vector_fp16(num_wf, num_wg, dev_wf_sz, operations);}
                         else if(v_dtype==BF16){run_metrics=kernel_launch_vector_bf16(num_wf, num_wg, dev_wf_sz, operations);}
                    }
                    else if (operations==M_WMMA)
                    {
                         if     (m_multype==FP64 && m_acctype==FP64)   {run_metrics=kernel_launch_wmma_f64_16x16x4_f64(num_wf, num_wg, dev_wf_sz);}
                         else if(m_multype==FP32 && m_acctype==FP32)   {run_metrics=kernel_launch_wmma_f32_16x16x4_f32(num_wf, num_wg, dev_wf_sz);}
                         else if(m_multype==TF32 && m_acctype==FP32)   {run_metrics=kernel_launch_wmma_f32_16x16x8_xf32(num_wf, num_wg, dev_wf_sz);}
                         else if(m_multype==FP16 && m_acctype==FP32)   {run_metrics=kernel_launch_wmma_f32_16x16x16_f16(num_wf, num_wg, dev_wf_sz);}
                         else if(m_multype==BF16 && m_acctype==FP32)   {run_metrics=kernel_launch_wmma_f32_16x16x16_bf16(num_wf, num_wg, dev_wf_sz);}
                         else if(m_multype==FP8 && m_acctype==FP32)    {run_metrics=kernel_launch_wmma_f32_16x16x32_fp8_fp8(num_wf, num_wg, dev_wf_sz);}
                         else if(m_multype==BF8 && m_acctype==FP32)    {run_metrics=kernel_launch_wmma_f32_16x16x32_bf8_bf8(num_wf, num_wg, dev_wf_sz);}
                         else if(m_multype==INT8 && m_acctype==INT32)  {run_metrics=kernel_launch_wmma_i32_16x16x32_i8(num_wf, num_wg, dev_wf_sz);}
                         else {std::cerr <<"[ERR!] Unsupported multiply/accumulation data type combinations!" << std::endl; exit(1);}
                    }
                    all_metrics.push_back(run_metrics);
               }
            }
        }
    }

    // Printing all metrics
    if (operations==V_ADD1 || operations==V_ADD2 || operations==V_MUL1 || operations==V_MUL2 || operations==V_FMA3 || operations==V_FMA2 || operations==V_FMA1)
    {
        std::cout << "Result " << str_operations << " with " << str_v_dtype << std::endl;
    }
    else if (operations==M_WMMA)
    {
        std::cout << "Result " << str_operations << " with " << str_m_multype <<"/"<< str_m_acctype << std::endl;
    }
    else if (operations==M_BLAS)
    {
        std::cout << "Result " << str_operations << " with " << str_m_multype <<"/"<< str_m_acctype<<"/"<< str_m_scaletype << std::endl;
    }

    metrics::print_csv_header();
    for(metrics item: all_metrics){
        item.print_csv();
    }

    
    return 0;
}