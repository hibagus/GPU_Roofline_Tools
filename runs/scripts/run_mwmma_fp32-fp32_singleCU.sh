#!/bin/bash

../../bin/amd_mi300x_bench \
    --device 7 \
    --operations M_WMMA \
    --matrix-mult-type fp32 \
    --matrix-accum-type fp32 \
    --min-wavefront 1 \
    --max-wavefront 16 \
    --step-wavefront 1 \
    --min-workgroup 1 \
    --max-workgroup 1 \
    --step-workgroup 1 \
    2>&1 | tee ../outputs/run_mwmma_fp32-fp32_singleCU.out