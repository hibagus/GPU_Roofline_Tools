#!/bin/bash

../../bin/amd_mi300x_bench \
    --device 0 \
    --operations M_BLAS \
    --matrix-mult-type fp64 \
    --matrix-accum-type fp64 \
    --matrix-scale-type fp64 \
    --dim-M 65536 \
    --dim-N 65536 \
    --dim-K 65536 \
    2>&1 | tee ../outputs/run_mblas_fp64-fp64-fp64.out