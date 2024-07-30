#!/bin/bash

../../bin/amd_mi300x_bench \
    --device 7 \
    --operations M_WMMA \
    --matrix-mult-type int8 \
    --matrix-accum-type int32 \
    --min-wavefront 1 \
    --max-wavefront 16 \
    --step-wavefront 1 \
    --min-workgroup 1 \
    --max-workgroup 1 \
    --step-workgroup 1 \
    2>&1 | tee ../outputs/run_mwmma_int8-int32_singleCU.out