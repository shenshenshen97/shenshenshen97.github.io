---
title: Tensorflow框架(1)张量 计算图 会话
layout: post
tags: tensorflow mooc 曹健
categories: 人工智能
---
中国大学mooc课程笔记：《人工智能实践：Tensorflow笔记》曹健 第三讲：Tensorflow框架 
[<u>课程地址</u>](https://www.icourse163.org/learn/PKU-1002536002?tid=1003797005#/learn/announce)


基于Tensorflow的NN： 用张量表示数据,用计算图搭建神经网络，用会话执行计算图，优化线上的权重（参数），得到模型。

# 张量

__张量（tensor）__：就是一个多维数组（列表） 阶：张量的维度 （可以表示0-n阶数组）

* 0阶 ----> 标量 scalar
* 1阶 ----> 向量 vector
* 2阶 ----> 矩阵 matrix
* n阶 ----> 张量 tensor 

判断张量是几阶的，就通过张量右边的方括号数，0 个是 0 阶，n 个是 n 阶，张量可以表示 0 阶到 n 阶数组（列表）

> 举例 t=[ [ [… ] ] ]为 3 阶

课程的实践环境是在Linux系统下，用vim编辑器。而我则是在windows下用PyCharm+Anaconda，来编写和执行文件。不过在这里还是记录下老师在Linux下的操作。

    vim ~/.vimrc 
    set ts=4  //设置一个TAB键位等效四个空格
    set nu    //在vim中显示行号
    
看具体代码实现一个__计算图__

    import tensorflow as tf
    a = tf.constant([1.0,2.0])    #定义一个张量等于[1.0,2.0]
    b = tf.constant([3.0,4.0])    #定义一个张量等于[3.0,4.0]
    result = a + b                #实现a+b的加法
    print result                  #打印出结果

可以打印出这样一句话：

    Tensor(“add:0”, shape=(2, ), dtype=float32)  
    //result 是一个名称为 add:0 的张量，shape=(2,)表示一维数组长度为 2，dtype=float32 表示数据类型为浮点型。
    

# 计算图

__计算图（Graph）__： 搭建神经网络的计算过程，是承载一个或多个计算节点的一张图，只搭建网络，不运算

举例：

神经网络的基本模型是神经元，神经元的基本模型其实就是数学中的乘、加运算。我们搭建如下的计算图： 

![QQ截图20190213105309.png](https://i.loli.net/2019/02/13/5c6386ae96402.png)

x1、x2 表示输入，w1、w2 分别是 x1 到 y 和 x2 到 y 的权重，y=x1*w1+x2*w2。 

我们实现上述计算图： 

    import tensorflow as tf           #引入模块 
    x = tf.constant([[1.0, 2.0]])     #定义一个 2 阶张量等于[[1.0,2.0]] 
    w = tf.constant([[3.0], [4.0]])   #定义一个 2 阶张量等于[[3.0],[4.0]] 
    y = tf.matmul(x, w)               #实现 xw 矩阵乘法 
    print y                           #打印出结果 

可以打印出这样一句话：

    Tensor(“matmul:0”, shape(1,1), dtype=float32）
    //y是一个名称为 matmul:0 的张量，表示两个长度为2的向量相乘，数据类型为浮点型

    
此时，只是完成了一个计算图的设计，就像是完成了公式的推导，还没有实际计算,如果我们想得到运算结果就要用到“__会话 Session()__”了。

# 会话

__会话（Session）__： 执行计算图中的节点运算。

我们用 with 结构实现，语法如下： 

    with tf.Session() as sess: 
    print sess.run(y)
    
举例：

对于刚刚所述计算图，我们执行 Session()会话可得到矩阵相乘结果：

    import tensorflow as tf           #引入模块 
    x = tf.constant([[1.0, 2.0]])     #定义一个 2 阶张量等于[[1.0,2.0]] 
    w = tf.constant([[3.0], [4.0]])   #定义一个 2 阶张量等于[[3.0],[4.0]] 
    y = tf.matmul(x, w)               #实现 xw 矩阵乘法 
    print y                           #打印出结果 
    with tf.Session() as sess: 
    print sess.run(y)            #执行会话并打印出执行后的结果 
    
可以打印出这样的结果：

    Tensor(“matmul:0”, shape(1,1), dtype=float32) 
    [[11.]]
    
我们可以看到，运行Session()会话前只打印出y是个张量的提示，运行Session()
会话后打印出了 y 的结果 1.0 x 3.0 + 2.0 x 4.0 = 11.0。

拓展：

计算两个2x2矩阵相乘：

![Image Title](https://quicklatex.com/cache3/6c/ql_1a7e0ec1e9202cb1020cf4fc0d79636c_l3.png)


{% highlight r %}

    import tensorflow as tf
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    w = tf.constant([[2.0, 1.0], [4.0, 3.0]])
    y = tf.matmul(x, w)
    print(y)
    with tf.Session() as sess:
        print(sess.run(y))
{% endhighlight %}

结果：是一个2x2矩阵

![result1.png](https://i.loli.net/2019/02/14/5c64cc08882a8.png)

但这在神经网络中似乎没有用，最开始应该是一个`1 x n`的<b>行向量</b>作为输入层，n为取得特征值数，然后经过隐藏层的计算，最后的结果为一个值。

![draw2.jpg](https://i.loli.net/2019/02/14/5c64d3fd22847.jpg)


*********************


小结：

这篇博客大部分时间用在了Jekyll writer的功能熟悉上，后边逐渐掌握软件功能后写博客的速度会逐渐加快，也希望能行成自己的博客风格。另外感叹自己线性代数学着忘着，得多写写博客反复理解吧。