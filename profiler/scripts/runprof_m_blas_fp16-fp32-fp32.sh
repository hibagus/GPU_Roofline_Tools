#!/bin/bash

rocprofv2 --input ../metrics/roofline.metrics \
          --plugin file \
          -d ../outputs \
          -o m_blas_fp16-fp32-fp32 \
          ../../runs/scripts/run_mblas_fp16-fp32-fp32.sh