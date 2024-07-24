#include <argparse/argparse.hpp>
#include <GPU_Roofline_Tools/launcher/rocm/kernel_launch_vector.hip.h>
#include <GPU_Roofline_Tools/launcher/rocm/kernel_device_init.hip.h>
#include <GPU_Roofline_Tools/utils/common/optype.h>
#include <GPU_Roofline_Tools/utils/common/ptype.h>
#include <GPU_Roofline_Tools/utils/common/metrics.h>

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
    program.add_argument("--iterations")
            .help("Number of iterations. Useful to eliminate any overheads since the operations of interest inside the kernel are repeated multiple times.")
            .scan<'i', int>()
            .default_value(1)
            .metavar("iter");
    program.add_argument("--operations")
            .help("Select operation types: V_ADD, V_MUL, V_FMA3, V_FMA2, V_FMA1, M_WMMA, M_BLAS")
            .default_value(std::string("V_FMA3"))
            .metavar("ops");

    // Vector Execution
    program.add_argument("--vector-data-type")
            .help("Select vector data type: fp64, fp32, fp16, or bf16. This is used for vector operations on Stream Processors (SP).")
            .default_value(std::string("fp16"))
            .metavar("v_dtype");

    // Matrix Execution
    program.add_argument("--matrix-ti-type")
            .help("Select matrix input data type: fp64, fp32, fp16, bf16, or int8. This is used for matrix operations on Matrix Cores (MC).")
            .default_value(std::string("fp16"))
            .metavar("m_titype");

    program.add_argument("--matrix-to-type")
            .help("Select matrix output data type: fp64, fp32, fp16, bf16, or int8. This is used for matrix operations on Matrix Cores (MC).")
            .default_value(std::string("fp16"))
            .metavar("m_totype");

    program.add_argument("--matrix-tc-type")
            .help("Select matrix computation data type: fp64, fp32, fp16, bf16, or int8. This is used for matrix operations on Matrix Cores (MC).")
            .default_value(std::string("fp16"))
            .metavar("m_tctype");

    


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
    // Iteration
    int num_iterations = program.get<int>("--iterations");
    // Data Type
    std::string str_v_dtype  = program.get<std::string>("--vector-data-type");
    std::string str_m_titype = program.get<std::string>("--matrix-ti-type");
    std::string str_m_totype = program.get<std::string>("--matrix-to-type");
    std::string str_m_tctype = program.get<std::string>("--matrix-tc-type");
    
    // Workgroup Control
    int min_workgroup  = program.get<int>("--min-workgroup");
    int max_workgroup  = program.get<int>("--max-workgroup");
    int step_workgroup = program.get<int>("--step-workgroup");

    // Wavefront Control
    int min_wavefront  = program.get<int>("--min-wavefront");
    int max_wavefront  = program.get<int>("--max-wavefront");
    int step_wavefront = program.get<int>("--step-wavefront");

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
    ptype m_titype;
    ptype m_totype;
    ptype m_tctype;

    // Data Type
    if       (str_v_dtype=="fp64"){v_dtype=FP64;}
    else if  (str_v_dtype=="fp32"){v_dtype=FP32;}
    else if  (str_v_dtype=="fp16"){v_dtype=FP16;}
    else if  (str_v_dtype=="bf16"){v_dtype=BF16;}
    else     {std::cerr <<"[ERR!] Argument parsing error: Unsupported vector data type!" << std::endl; exit(1);}

    if       (str_m_titype=="fp64"){m_titype=FP64;}
    else if  (str_m_titype=="fp32"){m_titype=FP32;}
    else if  (str_m_titype=="fp16"){m_titype=FP16;}
    else if  (str_m_titype=="bf16"){m_titype=BF16;}
    else if  (str_m_titype=="int8"){m_titype=INT8;}
    else     {std::cerr <<"[ERR!] Argument parsing error: Unsupported input matrix data type!" << std::endl; exit(1);}

    if       (str_m_totype=="fp64"){m_totype=FP64;}
    else if  (str_m_totype=="fp32"){m_totype=FP32;}
    else if  (str_m_totype=="fp16"){m_totype=FP16;}
    else if  (str_m_totype=="bf16"){m_totype=BF16;}
    else if  (str_m_totype=="int8"){m_totype=INT8;}
    else     {std::cerr <<"[ERR!] Argument parsing error: Unsupported output matrix data type!" << std::endl; exit(1);}

    if       (str_m_tctype=="fp64"){m_tctype=FP64;}
    else if  (str_m_tctype=="fp32"){m_tctype=FP32;}
    else if  (str_m_tctype=="fp16"){m_tctype=FP16;}
    else if  (str_m_tctype=="bf16"){m_tctype=BF16;}
    else if  (str_m_tctype=="int8"){m_tctype=INT8;}
    else     {std::cerr <<"[ERR!] Argument parsing error: Unsupported computation matrix data type!" << std::endl; exit(1);}

    // Operations
    optype operations;
    if       (str_operations=="V_ADD") {operations=V_ADD;}
    else if  (str_operations=="V_MUL") {operations=V_MUL;}
    else if  (str_operations=="V_FMA3"){operations=V_FMA3;}
    else if  (str_operations=="V_FMA2"){operations=V_FMA2;}
    else if  (str_operations=="V_FMA1"){operations=V_FMA1;}
    else if  (str_operations=="M_WMMA"){operations=M_WMMA;}
    else if  (str_operations=="M_BLAS"){operations=M_BLAS;}
    else     {std::cerr <<"[ERR!] Argument parsing error: Unsupported computation matrix data type!" << std::endl; exit(1);}
    
    // For storing the run metrics
    std::vector<metrics> all_metrics;

    // Workgroup Sweep
    for (int num_wg=min_workgroup; num_wg<=max_workgroup; num_wg=num_wg+step_workgroup)
    {
        // Wavefront Sweep
        for(int num_wf=min_wavefront; num_wf<=max_wavefront; num_wf=num_wf+step_wavefront)
        {
           std::cout << "[INFO] Running with " << num_wf << " wavefront(s) per workgroup and " << num_wg << " workgroup(s)" << std::endl; 
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
                if (operations==V_ADD || operations==V_MUL || operations==V_FMA3 || operations==V_FMA2 || operations==V_FMA1)
                {
                     if     (v_dtype==FP64){run_metrics=kernel_launch_vector_fp64(num_wf, num_wg, dev_wf_sz, num_iterations, operations);}
                     else if(v_dtype==FP32){run_metrics=kernel_launch_vector_fp32(num_wf, num_wg, dev_wf_sz, num_iterations, operations);}
                     else if(v_dtype==FP16){run_metrics=kernel_launch_vector_fp16(num_wf, num_wg, dev_wf_sz, num_iterations, operations);}
                     else if(v_dtype==BF16){run_metrics=kernel_launch_vector_bf16(num_wf, num_wg, dev_wf_sz, num_iterations, operations);}
                }
                else if (operations==M_WMMA)
                {
     
                }
                else if (operations==M_BLAS)
                {
     
                }
                all_metrics.push_back(run_metrics);
           }
        }
    }

    // Printing all metrics
    if (operations==V_ADD || operations==V_MUL || operations==V_FMA3 || operations==V_FMA2 || operations==V_FMA1)
    {
        std::cout << "Result " << str_operations << " with " << str_v_dtype << std::endl;
    }
    else if (operations==M_WMMA)
    {
        std::cout << "Result " << str_operations << " with " << str_m_titype <<"/"<< str_m_totype <<"/"<< str_m_tctype << std::endl;
    }
    else if (operations==M_BLAS)
    {
        // TODO: Write!!!!!!!!!!
    }

    metrics::print_csv_header();
    for(metrics item: all_metrics){
        item.print_csv();
    }

    
    return 0;
}