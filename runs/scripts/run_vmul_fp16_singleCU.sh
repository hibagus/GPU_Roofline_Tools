#!/bin/bash

../../bin/amd_mi300x_bench \
    --device 0 \
    --operations V_MUL \
    --vector-data-type fp16 \
    --min-wavefront 1 \
    --max-wavefront 16 \
    --step-wavefront 1 \
    --min-workgroup 1 \
    --max-workgroup 1 \
    --step-workgroup 1 \
    2>&1 | tee ../outputs/run_vmul_fp16_singleCU.out