#!/bin/bash

rocprofv2 --input ../metrics/roofline.metrics \
          --plugin file \
          -d ../outputs \
          -o m_blas_int8-int32-int32 \
          ../../runs/scripts/run_mblas_int8-int32-int32.sh