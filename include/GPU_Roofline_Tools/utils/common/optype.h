#pragma once

enum optype 
{
    V_ADD1, // A = A + constant
    V_ADD,  // A = A + B
    V_ADD2, // Packed Add A=A+B, C=C+B
    V_MUL,  // A = A * constant
    V_MUL2, // A = A * B
    V_FMA3, // A = A*B + C
    V_FMA2, // A = A*constant + B
    V_FMA1, // A = A*constant + A
    M_WMMA, // Matrix Multiplication using rocWMMA 
    M_BLAS  // Matrix Multiplication using rocBLAS
};