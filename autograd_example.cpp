#include<iostream>
#include<vector>
#include<memory>
#include<queue>
#include<cmath>

namespace autograd{
    enum Operator{
        NONE, ADD, MUL, SUB, DIV,
        LN, EXP, SIN, COS
    };
    inline float operate(float vl, float vr, Operator op){
        switch (op){
            case ADD: return vl + vr; case SUB: return vl - vr;
            case MUL: return vl * vr; case DIV: return vl / vr;
            default: throw std::invalid_argument("Unsupported operator!\n");
        }
    }
    inline float operate(float val, Operator op){
        switch (op){
            case LN: return log(val); 
            case EXP: return exp(val);
            case SIN: return sin(val);
            case COS: return cos(val);
            default:throw std::invalid_argument("Unsupported operator!\n");
        }
    }
    struct Node{
        float value;
        float grad;
        Node* left_node;
        Node* right_node;
        Operator op;
        Node(float value, float gard){
            this->value = value;
            this->grad = gard;
            this->left_node = nullptr;
            this->right_node = nullptr;
            this->op = NONE;
        }
        Node(Node* left, Node* right, Operator op){
            this->value = operate(left->value, right->value, op);
            this->grad = 0.0f;
            this->left_node = left;
            this->right_node = right;
            this->op = op;
        }
        Node(Node* single, Operator op){
            this->value = operate(single->value, op);
            this->grad = 0.0f;
            this->left_node = single;
            this->right_node = nullptr;
            this->op = op;
        }
        void accumulate(float delta_grad){
            this->grad += delta_grad;
        }
    };

    struct GradNode{
        GradNode(float value, float grad=0)
            :src(new Node(value, grad)){}
        GradNode(GradNode& left, GradNode& right, Operator op)
            :src(new Node(left.src, right.src, op)){}
        GradNode(GradNode& single, Operator op)
            :src(new Node(single.src, op)){}
        ~GradNode(){delete src;}
        void backward(){
            std::queue<Node*> process_queue;
            src->grad = 1.0f;
            process_queue.push(src);
            while (!process_queue.empty()){
                Node* it = process_queue.front();
                process_queue.pop();
                Node* l = it->left_node;
                Node* r = it->right_node;
                if(it->op == NONE) continue;
                switch (it->op){
                    case ADD: l->accumulate(it->grad), r->accumulate(it->grad); break;
                    case SUB: l->accumulate(it->grad), r->accumulate(-it->grad); break;
                    case MUL: l->accumulate(r->value * it->grad), r->accumulate(l->value * it->grad); break;
                    case DIV: l->accumulate(it->grad / r->value), r->accumulate(-it->grad * l->value / (r->value * r->value)); break;
                    case LN: l->accumulate(it->grad / l->value); break;
                    case EXP: l->accumulate(it->grad * exp(l->value)); break;
                    case SIN: l->accumulate(it->grad * cos(l->value)); break;
                    case COS: l->accumulate(-it->grad * sin(l->value)); break;
                    default: throw std::invalid_argument("Unsupported operator!\n");
                }
                if(l !=nullptr)process_queue.push(l);
                if(r !=nullptr)process_queue.push(r);
            }
        }
        float value(){return src->value;}
        float grad(){return src->grad;}
    private:
        Node* src; 
    };
}

using namespace std;
using namespace autograd;

int main(){
    GradNode x(exp(1));
    GradNode x2(x, x, MUL);
    GradNode ln(x2, LN);

    ln.backward();
    std::cout<<x.grad();
}