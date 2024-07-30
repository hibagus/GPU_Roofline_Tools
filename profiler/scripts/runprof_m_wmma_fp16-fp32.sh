#!/bin/bash

rocprofv2 --input ../metrics/m_wmma_fp16-fp32.metrics \
          --plugin file \
          -d ../outputs/ \
          -o m_wmma_fp16-fp32 \
          ../../runs/scripts/run_mwmma_fp16-fp32_singleCU.sh