#pragma once

enum optype 
{
    V_ADD,
    V_ADD2, // Packed Add A=A+B, C=C+B
    V_MUL,
    V_FMA3, // A = A*B + C
    V_FMA2, // A = A*constant + B
    V_FMA1, // A = A*constant + A
    M_WMMA, // Matrix Multiplication using rocWMMA 
    M_BLAS  // Matrix Multiplication using rocBLAS
};