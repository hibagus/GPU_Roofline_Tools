#!/bin/bash

../../bin/amd_mi300x_bench \
    --device 0 \
    --operations V_FMA3 \
    --vector-data-type fp32 \
    --min-wavefront 1 \
    --max-wavefront 16 \
    --step-wavefront 1 \
    --min-workgroup 1 \
    --max-workgroup 304 \
    --step-workgroup 1 \
    2>&1 | tee ../outputs/run_fma3_fp32_gpuWide.out