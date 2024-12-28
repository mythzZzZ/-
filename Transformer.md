# Attention Is All You Need

RNN

- 时序模型，把以前的信息放到隐藏状态里面记录（ht）里面，并行度比较差，计算第n个状态需要计算完前n个状态。如果信息很长，计算到后面信息时，前面的信息可能会发生丢失



**RNN缺点，按照时序来做计算，Transformer不需要时序运算 并行度更好**



把长度为m的句子每个单词拿出来，每一个词用向量来表示（向量被encoder识别），向量进入encoder后生成长度为n的序列（可以与输入不一样）。解码的时候词一个个生成（自回归），生成的时候是一个个词的输出，前面生成输出的词语是后面生成输出的词语的输入（自回归）



![image-20240430164724528](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240430164724528.png)



## LN

- 蓝色是BN（把样本变成均值为0，方差为1，不同的卷积位置来做均值和方差）
- 黄色是LN，每个样本计算自己的均值和方差（为什么不用BN，因为BN计算的是不同样本的均值和方差，当样本长度不一样的时候（有一些样本特别长），得到的结果不那么好用，所以换成LN，计算样本自己的均值和方差）

![image-20240430171334133](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240430171334133.png)



## Embedding

-  把单词映射成序列



## 自注意力机制

- 一个输入变成向量之后加入位置编码，然后复制三份（q,k,v是同一个东西用来训练）
- 输入和输出的大小是同一个东西（输出是value的加权和，权重来自于Q和K的内积）
- 三个东西都一样，模型有啥用
  - 在进行多头注意力的时候，是先对qkv进行投影在进行注意力的，关键在于投影，通过投影学习到不同的距离空间出来



![image-20240430175436514](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240430175436514.png)



## 多头注意力（8个头）

- 把QKV通过linear层投影到低纬度，在低纬度上有多个注意力头，把QKV的投影在不同的头做注意力运算，然后在通过投影拼接回去 （有CNN多个输出通道的感觉）
  - 为什么要做投影，因为普通的注意力头没有可学习的参数，通过linear投影来学习东西

![image-20240430174644409](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240430174644409.png)

## 注意力函数

- 注意力函数将一个query和key-value对映射成一个输出的一个函数（query，key，value都是一个向量），output是value的加权和，输出的维度和value的维度是一样的。（权重怎么来的）权重是key 和 query的相似度算来的（注意力机制算法）



![image-20240430172035098](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240430172035098.png)





## Transformer的注意力

- QK做向量内积，计算相似度，内积为1相似度最大，使用点积注意力机制 在除以一个分母（为什么要除分母dk，当向量长度太长的时候做点积，相对差距就会变大，softmax得到的值更加靠近1，剩下的值靠近于0（向两端靠拢），此时梯度很小（因为我们的值靠近1了不用跑了）。这时候我们除以dk可以更好的反向传播）

![image-20240430172251920](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240430172251920.png)

**对Q，K进行矩阵运算，得到新的矩阵每一行对应V的权重，在乘以V矩阵得到最后的输出**



![image-20240430172546093](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240430172546093.png)





## **Feed Forward**

- 相当与MLP，对每一个词进行MLP
  - 膨胀四倍激活，然后投影回原来大小



## Masker-Multi-Head的作用

- 保证输入进来的时候，t时间不会看到t时间以后的那些输入（保证训练和预测的行为一致）
- mask：假如query和key是等长的，且在时间上能对应起来，在第t时间，我们应该看到的是kt之前的东西，**使用kt之前的东西来运算**（计算的时候已经全部算出来了，但是不让他看到）
  - mask是如何实现隐藏kt之后的东西，给kt之后的东西一个很大的负数，这个负数经过softmax之后就变成权重0了



## Masked Multi-head attention

- Masked Multi-head attention那一边的attention是用来test的
  - 输入要查询的句子，分成QKV，此时输入的句子经过Masked Multi-head attention变成Q，与已经训练好的左边已经训练好的KV进行 QK注意力查询，得到V的权重输出



![image-20240430180419019](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240430180419019.png)







## 其他注意力机制

- 加型注意力机制（处理query与key不等长的情况）
- 点积注意力机制（和Transformer一样，Transformer多一个分母）





## Transformer与RNN的区别

- RNN是上一个值的输出是下一个值的输入 来传递历史信息
- Transformer通过在输入加入位置编码（Positional Encoding）来处理时序信息，这个编码是如何算的？（使用周期不一样的sin cons函数来算出来的，编码值在-1 到+1 之间）













# 大语言模型

BERT

- 它使用大量未标记的文本进行预训练，然后使用标记的数 据进行微调
