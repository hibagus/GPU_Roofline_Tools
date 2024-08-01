#!/bin/bash

../../bin/amd_mi300x_bench \
    --device 0 \
    --operations M_BLAS \
    --matrix-mult-type bf16 \
    --matrix-accum-type bf16 \
    --matrix-scale-type fp32 \
    --dim-M 65536 \
    --dim-N 65536 \
    --dim-K 65536 \
    2>&1 | tee ../outputs/run_mblas_bf16-bf16-fp32.out