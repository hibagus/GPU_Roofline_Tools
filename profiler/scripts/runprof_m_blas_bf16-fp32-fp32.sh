#!/bin/bash

rocprofv2 --input ../metrics/roofline.metrics \
          --plugin file \
          -d ../outputs \
          -o m_blas_bf16-fp32-fp32 \
          ../../runs/scripts/run_mblas_bf16-fp32-fp32.sh