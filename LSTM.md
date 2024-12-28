

# LSTM

https://zhuanlan.zhihu.com/p/32085405



## RNN是什么

循环神经网路

- 时序模型，专门处理序列信息的，把以前的信息放到隐藏状态里面记录（ht）里面持续传递，并行度比较差，计算第n个状态需要计算完前n个状态（第n-1个状态是第n个状态的输入）。如果信息很长，计算到后面信息时，前面的信息可能会发生丢失
- 做乘法计算 往前走

- **RNN中常用的激活函数是tanh或sigmoid函数，这些函数的导数在[-1, 1]或[0, 1]的范围内，导致梯度在传播过程中可能会指数级地衰减或增长。**

![image-20240501182545868](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240501182545868.png)

![image-20240501182424840](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240501182424840.png)





## LSTM是什么

长短期记忆网络，一种特殊的RNN，设计用来解决传统RNN中梯度消失（vanishing gradient）和梯度爆炸（exploding gradient）。处理长序列数据时比RNN更优秀，RNN容易丢失信息

- **传统的RNN在处理长序列数据时会出现梯度消失或梯度爆炸的问题，导致模型难以捕获长期依赖关系。LSTM通过使用门控机制（更多的参数来控制参数，输入门、遗忘门和输出门），能够有选择地记忆或遗忘输入数据中的信息，从而更好地处理长序列数据**



![image-20240501184515849](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240501184515849.png)

![image-20240501184531395](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240501184531395.png)

