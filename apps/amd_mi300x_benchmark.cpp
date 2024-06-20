#include <argparse/argparse.hpp>
#include <GPU_Roofline_Tools/launcher/rocm/kernel_launch.hip.h>
#include <GPU_Roofline_Tools/utils/common/optype.h>

int main(int argc, char *argv[])
{
    kernel_launch_fp64(1024, 1, V_ADD);
    return 0;
}