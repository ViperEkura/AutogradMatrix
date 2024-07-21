#include<iostream>
#include"auto_grad/AutoGrad.h"
using namespace std;


int main(){
    GradMatrix w0(784 * 100); w0.rand(0, 0.03);
    GradMatrix b0(1 * 100); b0.rand(0, 0.3);
    GradMatrix w1(100 * 1); w1.rand(0, 0.3);
    GradMatrix b1(1); b1.rand();
    GradMatrix a(1 * 784, false); a.rand();
    GradMatrix tgt(1, false); tgt.rand();
    float lr = 0.0005;

    for(int i=0;i<10; ++i){
        GradMatrix t0(a, w0, MATMUL,1, 784, 100); // (1, 10)
        GradMatrix d0(t0, b0, ADD);
        GradMatrix t1(d0, w1, MATMUL, 1, 100, 1); // (1, 1)
        GradMatrix d1(t1, b1, ADD);
        GradMatrix loss(d1, d1, MUL);
        loss.backward();
        w0.step(lr); b0.step(lr); w1.step(lr);b1.step(lr);
        printf("loss:\t%f\n", loss.node->value.src.ptr[0]);
    }
}