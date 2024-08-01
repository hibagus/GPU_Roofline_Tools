#!/bin/bash

../../bin/amd_mi300x_bench \
    --device 0 \
    --operations M_BLAS \
    --matrix-mult-type int8 \
    --matrix-accum-type int32 \
    --matrix-scale-type int32 \
    --dim-M 65536 \
    --dim-N 65536 \
    --dim-K 65536 \
    2>&1 | tee ../outputs/run_mblas_int8-int32-int32.out