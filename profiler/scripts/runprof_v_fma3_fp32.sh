#!/bin/bash

rocprofv2 --input ../metrics/v_fma3_fp32.metrics \
          --plugin file \
          -d ../outputs \
          -o v_fma3_fp32 \
          ../../runs/scripts/run_vfma3_fp32_singleCU.sh