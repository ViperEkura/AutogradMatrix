#ifndef OPERATOR_H
#define OPERATOR_H
#include "Matrix.h"
#include <stdexcept>
enum Operator{
    NONE, ADD, SUB, MUL, DIV, 
    MATMUL, EXP, RELU
};

inline Matrix exp(const Matrix& val){
    Matrix copy_val = val.copy();
    for(int i=0;i<copy_val.capacity;++i){
        copy_val.src.ptr[i] = std::exp(copy_val.src.ptr[i]);
    }
    return copy_val;
}

inline Matrix matmul(const Matrix& a, const Matrix& b, int m, int n, int p){
    Matrix output(m * p);
    output.zero();
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < p; ++k) {
                output.src.ptr[i * p + k] += a.src.ptr[i * n + j] * b.src.ptr[j * p + k];
            }
        }
    }
    return output;
}

inline Matrix relu(const Matrix& val){
    Matrix relu_val = val.copy();
    for(int i=0;i<relu_val.capacity;++i){
        if(relu_val.src.ptr[i]< 0) relu_val.src.ptr[i] = 0;
    }
    return relu_val;
}

inline Matrix dr_relu(const Matrix& val, Matrix& grad){
    for(int i=0;i<val.capacity;++i){
        if(val.src.ptr[i] < 0) grad.src.ptr[i] = 0;
    }
    return grad;
}


Matrix opearte(const Matrix& a, const Matrix& b, Operator op){
    switch (op){
        case ADD: return a + b; case SUB: return a - b;
        case MUL: return a * b; case DIV: return a / b;
        default: throw std::invalid_argument("Operator!");
    }
}

Matrix opearte(const Matrix& a, Operator op){
    switch (op){
        case EXP: return exp(a);
        case RELU: return relu(a);
        default: throw std::invalid_argument("Operator!");
    }
}

Matrix opearte(const Matrix& a, const Matrix& b, int m, int n , int p){
    return matmul(a, b, m, n, p);
}

#endif