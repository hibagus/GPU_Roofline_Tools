#include <iostream>
#include <GPU_Roofline_Tools/utils/common/metrics.h>

void metrics::print_csv()
{
    std::cout 
    << n_iter      <<", "
    << n_thread    <<", "
    << n_wg        <<", "
    << n_wf        <<", "
    << wf_size     <<", "
    << wg_size     <<", "
    << n_flops     <<", "
    << n_bytes     <<", "
    << time_ms     <<", "
    << avg_clock   <<", "
    << max_clock   <<", "
    << min_clock   <<", "
    << stdev_clock <<" "
    << std::endl; 
}

void metrics::print_csv_header()
{
    std::cout 
    << "n_iter"      <<", "
    << "n_thread"    <<", "
    << "n_wg"        <<", "
    << "n_wf"        <<", "
    << "wf_size"     <<", "
    << "wg_size"     <<", "
    << "n_flops"     <<", "
    << "n_bytes"     <<", "
    << "time_ms"     <<", "
    << "avg_clock"   <<", "
    << "max_clock"   <<", "
    << "min_clock"   <<", "
    << "stdev_clock" <<" "
    << std::endl;     
}
