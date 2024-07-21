#ifndef AUTO_GRAD_H
#define AUTO_GRAD_H

#include <queue>
#include "Matrix.h"
#include "Operator.h"

struct BaseMatrixNode{
    Matrix value;
    Matrix grad;
    BaseMatrixNode* left;
    BaseMatrixNode* right;
    Operator op;
    int m, n ,p;
    bool require_grad;

    BaseMatrixNode():left(nullptr),right(nullptr),op(NONE){}
    BaseMatrixNode(int capacity, bool require_grad=false)
        :left(nullptr),right(nullptr),op(NONE){
        value = Matrix(capacity);
        this->require_grad = require_grad;
        if(require_grad){
            grad = Matrix(capacity);
            grad.zero();
        }
    }
    BaseMatrixNode(BaseMatrixNode* left, BaseMatrixNode* right, Operator op, bool require_grad){
        this->left = left;
        this->right = right;
        this->op = op;
        value = opearte(left->value, right->value, op);
        this->require_grad = require_grad;
        if(require_grad){
            grad = Matrix(value.capacity);
            grad.zero();
        }
    }
    BaseMatrixNode(BaseMatrixNode* left, BaseMatrixNode* right, Operator op , int m , int n, int p, bool require_grad){
        this->left = left;
        this->right = right;
        this->op = op;
        this->m = m, this->n = n, this->p = p;
        value = opearte(left->value, right->value, m, n ,p);
        this->require_grad = require_grad;
        if(require_grad){
            grad = Matrix(value.capacity);
            grad.zero();
        }
    }
    BaseMatrixNode(BaseMatrixNode* single, Operator op, bool require_grad){
        this->left = single;
        this->right =nullptr;
        this->op = op;
        value = opearte(single->value, op);
        this->require_grad = require_grad;
        if(require_grad){
            grad = Matrix(value.capacity);
            grad.zero();
        }
    }
    void accumulate(const Matrix& delta_grad){
        grad += delta_grad;
    }
};

struct GradMatrix{
    BaseMatrixNode* node;
    GradMatrix(int capacity, bool require_grad=true)
        :node(new BaseMatrixNode(capacity, require_grad)){}
    GradMatrix(const GradMatrix& left, const GradMatrix& right, Operator op, bool require_grad=true)
        :node(new BaseMatrixNode(left.node, right.node, op, require_grad)){}
    GradMatrix(const GradMatrix& left, const GradMatrix& right, Operator op, int m , int n, int p, bool require_grad=true)
        :node(new BaseMatrixNode(left.node, right.node, op, m, n, p, require_grad)){}
    GradMatrix(const GradMatrix& single, Operator op, bool require_grad=true)
        :node(new BaseMatrixNode(single.node, op, require_grad)){}
    ~GradMatrix(){delete node;}
    
    void backward(){
        std::queue<BaseMatrixNode*> process_queue;
        node->grad.fill(1.0f);
        process_queue.push(node);
        while (!process_queue.empty()){
            BaseMatrixNode* it = process_queue.front();
            process_queue.pop();
            BaseMatrixNode* l = it->left;
            BaseMatrixNode* r = it->right;  
            if(it->op == NONE) continue;
            if(l->require_grad) switch (it->op){
                case ADD: l->accumulate(it->grad); break;
                case SUB: l->accumulate(it->grad); break;
                case MUL: l->accumulate(r->value * it->grad); break;
                case DIV: l->accumulate(it->grad / r->value); break;
                case EXP: l->accumulate(it->grad * exp(l->value)); break;
                case MATMUL: l->accumulate(matmul(it->grad, r->value.transpose(it->n, it->p), it->m, it->p, it->n)); break;
                case RELU: l->accumulate(dr_relu(l->value, it->grad)); break;
                default: throw std::invalid_argument("unsuported operator!");
            }

            if(r->require_grad) switch (it->op){
                case ADD: r->accumulate(it->grad); break;
                case SUB: r->accumulate((it->grad.copy()).neg()); break;
                case MUL: r->accumulate(l->value * it->grad); break;
                case DIV: r->accumulate((it->grad * l->value / (r->value * r->value)).neg()); break;
                case MATMUL: r->accumulate(matmul(r->value.transpose(it->m, it->n), it->grad, it->n, it->m, it->p)); break;
                default: throw std::invalid_argument("unsuported operator!");
            }
            if(l !=nullptr)process_queue.push(l);
            if(r !=nullptr)process_queue.push(r);
        }
    }
    void rand(float mean=0,float std=1){node->value.rand(mean, std);}
    void zero(){node->value.zero();}
    void fill(float val){node->value.fill(val);}
    void step(float lr=0.2f){
        node->value -= node->grad * lr;
        node->grad.zero();
    }
    Matrix grad(){return node->grad;}
    Matrix value(){return node->value;}

};

#endif