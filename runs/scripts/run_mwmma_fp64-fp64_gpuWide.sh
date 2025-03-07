#!/bin/bash

../../bin/amd_mi300x_bench \
    --device 7 \
    --operations M_WMMA \
    --matrix-mult-type fp64 \
    --matrix-accum-type fp64 \
    --min-wavefront 16 \
    --max-wavefront 16 \
    --step-wavefront 1 \
    --min-workgroup 1 \
    --max-workgroup 304 \
    --step-workgroup 1 \
    2>&1 | tee ../outputs/run_mwmma_fp64-fp64_gpuWide.out