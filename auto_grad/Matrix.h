#ifndef MATRIX_H
#define MATRIX_H
#include<random>
#include<chrono>

struct MatrixSrc{
public:
    float* ptr;
    int* ref_cnt;
public:
    MatrixSrc():ptr(nullptr), ref_cnt(nullptr){};
    MatrixSrc(int size):ptr(new float[size]), ref_cnt(new int(1)){};
    MatrixSrc(const MatrixSrc& other){
        ptr = other.ptr;
        ref_cnt = other.ref_cnt;
        if(other.ref_cnt != nullptr) ++(*other.ref_cnt);
    }
    MatrixSrc operator=(const MatrixSrc& other){
        if(&other == this) return *this;
        if(ptr != nullptr){
            if(--(*ref_cnt) == 0)
                delete ref_cnt, delete[] ptr;
        }
        ptr = other.ptr;
        ref_cnt = other.ref_cnt;
        if(other.ref_cnt != nullptr) ++(*other.ref_cnt);
        return *this;
    }
    ~MatrixSrc(){
        if(ptr == nullptr) return;
        if(--(*ref_cnt) != 0) return;
        delete ref_cnt, delete[] ptr;
        ref_cnt = nullptr, ptr = nullptr;
    }
};

struct Matrix{
public:
    MatrixSrc src;
    int capacity;
public:
    Matrix():capacity(0){}
    Matrix(int capacity){
        this->capacity = capacity;
        src = MatrixSrc(capacity);
    }
    Matrix(const Matrix& other){
        capacity = other.capacity;
        src = other.src;
    }
    Matrix operator=(const Matrix& other){
        capacity = other.capacity;
        src = other.src;
        return *this;
    }
    Matrix operator+=(const Matrix& other){
        for(int i=0;i<capacity;++i)
            src.ptr[i] += other.src.ptr[i];
        return *this;
    }
    Matrix operator-=(const Matrix& other){
        for(int i=0;i<capacity;++i)
            src.ptr[i] -= other.src.ptr[i];
        return *this;
    }
    Matrix operator+(const Matrix& other)const{
        Matrix tgt(capacity);
        for(int i=0;i<capacity;++i)
            tgt.src.ptr[i] = src.ptr[i] + other.src.ptr[i];
        return tgt;
    }
    Matrix operator-(const Matrix& other)const{
        Matrix tgt(capacity);
        for(int i=0;i<capacity;++i)
            tgt.src.ptr[i] = src.ptr[i] - other.src.ptr[i];
        return tgt;
    }
    Matrix operator*(const Matrix& other)const{
        Matrix tgt(capacity);
        for(int i=0;i<capacity;++i)
            tgt.src.ptr[i] = src.ptr[i] * other.src.ptr[i];
        return tgt;
    }
    Matrix operator*(const float val)const{
        Matrix tgt(capacity);
        for(int i=0;i<capacity;++i)
            tgt.src.ptr[i] = src.ptr[i] * val;
        return tgt;
    }
    Matrix operator/(const Matrix& other)const{
        Matrix tgt(capacity);
        for(int i=0;i<capacity;++i)
            tgt.src.ptr[i] = src.ptr[i] / other.src.ptr[i];
        return tgt;
    }
    Matrix copy() const{
        Matrix tgt(capacity);
        for(int i=0;i<capacity;++i)
            tgt.src.ptr[i] = src.ptr[i];
        return tgt;
    }
    void rand(float mean=0, float std=1){
        std::chrono::system_clock::time_point now = 
            std::chrono::system_clock::now();
        std::chrono::milliseconds timestamp = 
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
        unsigned int seed = (unsigned int)(timestamp.count());
        std::mt19937 engine(seed);
        std::normal_distribution<float> distribution(mean, std);
        for(int i=0;i<capacity;++i)
            src.ptr[i] = distribution(engine);
    }
    void zero(){
        std::fill(src.ptr, src.ptr + capacity, 0.0f);
    }
    void fill(float val){
        std::fill(src.ptr, src.ptr + capacity, val);
    }
    Matrix neg(){
        for(int i=0;i<capacity;++i)
            src.ptr[i] = -src.ptr[i];
        return *this;
    }
    Matrix transpose(int m, int n){
        Matrix new_mat(m * n);
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                new_mat.src.ptr[j*m + i] = src.ptr[i*n + j];
            }
        }
        return new_mat;
    }
};

#endif