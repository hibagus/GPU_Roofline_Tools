#pragma once

enum optype 
{
    V_ADD,
    V_MUL,
    V_FMA3, // A = A*B + C
    V_FMA2, // A = A*constant + B
    V_FMA1  // A = A*constant + A
};