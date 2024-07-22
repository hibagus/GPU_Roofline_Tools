#pragma once

enum ptype 
{
    BF16V, // Vector-bound Operation on BF16
    FP16V, // Vector-bound Operation on FP16
    FP32V, // Vector-bound Operation on FP32
    FP64V, // Vector-bound Operation on FP64
    FP16M, // Matrix-bound Operation on FP16
    FP32M  // Matrix-bound Operation on FP32
};