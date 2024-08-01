#!/bin/bash

rocprofv2 --input ../metrics/roofline.metrics \
          --plugin file \
          -d ../outputs \
          -o m_blas_fp64-fp64-fp64 \
          ../../runs/scripts/run_mblas_fp64-fp64-fp64.sh