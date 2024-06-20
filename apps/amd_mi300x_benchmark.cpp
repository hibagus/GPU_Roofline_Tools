#include <argparse/argparse.hpp>
#include <GPU_Roofline_Tools/launcher/rocm/kernel_launch.hip.h>

int main(int argc, char *argv[])
{
    kernel_launch_fp64(1024, 1);
    kernel_launch_fp64(1024, 2);
    kernel_launch_fp64(1024, 3);
    kernel_launch_fp64(1024, 4);
    kernel_launch_fp64(1024, 5);
    kernel_launch_fp64(1024, 6);
    kernel_launch_fp64(1024, 7);
    return 0;
}