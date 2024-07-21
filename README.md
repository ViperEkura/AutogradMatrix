# AutogradMatrix

### 简介

这是一个简单的自动微分系统，通过引用计数维护一个类似二叉树的计算图结构，当引用次数为0的时候自动删除节点，维护计算图结构

使用类似torch的backward方法

### 1.创建节点 

```c++
GradMatrix a(capacity);         //创建大小为capacity的矩阵
GradMatrix a(capcacity, false); // 创建大小为capacity的矩阵, 而且不更新梯度
GradMatrix a(a, b, OP);          // 根据operator创建节点， 默认更新梯度
GradMatrix a(a, b, MATMUL, m, n, p);  // (m, n)大小，(n ,p)大小的矩阵之间的矩阵乘
GradMatrix a(b, OP);             // 支持单元素操作符
```



### 2. 初始化

```c++
a.fill(val);      //全部填充为val
a.rand(mean,std); // 正态分布填充, 默认mean=0, std=1
a.zero();		  // 全部填充为0
```



### 3.计算梯度

```c++
loss.backward(); // 反向传播
weight.step(lr); // 以lr作为学习率更新
```

### 4. 使用示例

```c++
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
```





