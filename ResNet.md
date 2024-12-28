ResNet

-  论文大意是当神经网络越来越深，在反向传播时候，反传回来的梯度之间的相关性会越来越差，最后接近白噪声。如果梯度接近白噪声，那梯度更新可能根本就是在做随机扰动。理论上这种相关性的衰减越少，说明反向传播回来的信息的丢失/损失越少
- y = fx + x

- 在深层网络中，需要通过恒等映射来减少梯度相关性的衰减，如果通过 y = fx来学习恒等映射是比较困难的，所以就有了y = fx + x
  - x直接加过来，让fx = 0更简单
- 同时解决梯度消失的问题
  - 前向传播函数 y = f(x) + x ，你会发现求导结果是 1 + f'(x)，也就是说无论f'(x)多么的小，因为1的存在，链式求导的结果不会为0，进而解决了梯度消失的问题。



Inception

https://zhuanlan.zhihu.com/p/30756181

![image-20240508191027037](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240508191027037.png)

![image-20240508191032291](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240508191032291.png)