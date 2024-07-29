#!/bin/bash

if [ $# -lt 2 ]
  then
    echo "No arguments supplied."
    echo "Usage: $0 [GPU_Index] [Sampling Interval (Seconds)]"
    exit 1
fi

OUTPUT_PATH="../outputs/"
mkdir -p $OUTPUT_PATH

amd-smi metric --csv --file ../outputs/$3 -g $1 --watch $2 --power --clock --temperature