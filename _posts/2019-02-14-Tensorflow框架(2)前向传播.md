---
title: Tensorflow框架(2)前向传播
layout: post
tags: tensorflow mooc 曹健
categories: 人工智能
---
中国大学mooc课程笔记：《人工智能实践：Tensorflow笔记》曹健 第三讲：Tensorflow框架 
[<u>课程地址</u>](https://www.icourse163.org/learn/PKU-1002536002?tid=1003797005#/learn/announce)

基于Tensorflow的NN： 用张量表示数据,用计算图搭建神经网络，用会话执行计算图，优化线上的权重（参数），得到模型。

### 神经网络的参数

<b>神经网络的参数</b>：是指神经元线上的权重 w，用变量表示，一般会先随机生成 
这些参数。生成参数的方法是让w等于tf.Variable，把生成的方式写在括号里。

神经网络中常用的生成随机数/数组的函数有：

|   代码  |  效果   |
|:---:|:---:|
|  tf.random_normal()     |    生成正态分布随机数    |
|   tf.truncated_normal()    |      生成去掉过大偏离点的正态分布随机数  |
|  tf.random_uniform()     |      生成均匀分布随机数   |
|   tf.zeros     |    表示生成全 0 数组     |
|   tf.ones     |     表示生成全 1 数组     |
|   tf.fill   |   表示生成全定值数组    |
|  tf.constant   |    表示生成直接给定值的数组   |

举例：

① w=tf.Variable(tf.random_normal([2,3],stddev=2, mean=0, seed=1))，表
示生成正态分布随机数，形状两行三列，标准差是 2，均值是 0，随机种子是 1。如果去掉<b>随机种子</b>每次生成的随机数将不一致。

② w=tf.Variable(tf.Truncated_normal([2,3],stddev=2, mean=0, seed=1))，
表示去掉偏离过大的正态分布，也就是如果随机出来的数据偏离平均值超过两个
标准差，这个数据将重新生成。 

③ w=random_uniform(shape=7,minval=0,maxval=1,dtype=tf.int32，seed=1),
表示从一个均匀分布[minval maxval)中随机采样，注意定义域是左闭右开，即
包含 minval，不包含 maxval。

④ 除了生成随机数，还可以生成常量。tf.zeros([3,2],int32)表示生成
[[0,0],[0,0],[0,0]]；tf.ones([3,2],int32)表示生成[[1,1],[1,1],[1,1]；
tf.fill([3,2],6)表示生成[[6,6],[6,6],[6,6]]；tf.constant([3,2,1])表示
生成[3,2,1]。<br>

### 神经网络的搭建

![flow1.png](https://i.loli.net/2019/02/14/5c6514a4341f1.png)

由此可见，基于神经网络的机器学习主要分为两个过程，即训练过程和使用过程。  
训练过程是第一步、第二步、第三步的循环迭代，使用过程是第四步，一旦参数
优化完成就可以固定这些参数，实现特定应用了。 

很多实际应用中，我们会先使用现有的成熟网络结构，喂入新的数据，训练相应
模型，判断是否能对喂入的从未见过的新数据作出正确响应，再适当更改网络结
构，反复迭代，让机器自动训练参数找出最优结构和参数，以固定专用模型。 

### 向前传播

<b>前向传播</b>就是搭建模型的计算过程，让模型具有推理能力，可以针对一组输入
给出相应的输出。

举例：

假如生产一批零件，体积为 x1，重量为 x2，体积和重量就是我们选择的特征，
把它们喂入神经网络，当体积和重量这组数据走过神经网络后会得到一个输出。
 
假如输入的特征值是：体积 0.7  重量 0.5 

![flow2.png](https://i.loli.net/2019/02/14/5c651589f2b83.png)

由搭建的神经网络可得，隐藏层节点 a11=x1* w11+x2*w21=0.14+0.15=0.29，同
理算得节点 a12=0.32，a13=0.38，最终计算得到输出层 Y=-0.015，这便实现了
前向传播过程。

推导：

![prt3.png](https://i.loli.net/2019/02/15/5c660069ea75c.png)

![prt4.png](https://i.loli.net/2019/02/15/5c6604229e236.png)

具体代码：


```python
# 两层简单神经网络（全连接）
import tensorflow as tf

# 定义 输入和参数
x = tf.constant([[0.7, 0.5]])
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义向前传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("y in tf3.py is :\n", sess.run(y))
```

结果：

![result2.png](https://i.loli.net/2019/02/15/5c6606987c772.png)

<b>喂入多组数据</b>：

用 placeholder 实现输入定义（sess.run 中喂入多组数据）的情况 
第一组喂体积 0.7、重量 0.5，第二组喂体积 0.2、重量 0.3，第三组喂体积 0.3 、
重量 0.4，第四组喂体积 0.4、重量 0.5. 

```
# 两层简单神经网络（全连接）
import tensorflow as tf

# 定义 输入和参数
# 用placeholder定义输入（sess.run喂多组数据）
x = tf.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义向前传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("the result of the tf3_3.py is :\n", sess.run(y, feed_dict={x:[[0.7, 0.5], [0.2, 0.3],[0.3, 0.4],[0.4, 0.5]]}))
    print("w1\n", sess.run(w1))
    print("w1\n", sess.run(w2))
```

![result3.png](https://i.loli.net/2019/02/15/5c6609d3608b9.png)


--------------------
前向传播过程是通过输入特征计算出y值的过程。输入层一定是一行N列的一组或多组数据，可隐藏层的维度如何确定的呢？输出层必须为一个值吗？输出一个一行N列的矩阵是否有意义？