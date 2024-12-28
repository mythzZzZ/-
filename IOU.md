# IOU



https://blog.csdn.net/neil3611244/article/details/113794197

![image-20240428160504941](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428160504941.png)

![image-20240428150623109](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428150623109.png)



# GIOU

https://zhuanlan.zhihu.com/p/374398128

**IOU存在的问题**

- 知道GT与anchor的交并比，但是不能放映出GT与anchor的距离。当两个框距离很远时，loss等于0，无法进一步学习训练
- 预测框和真是框无法反映重合度的大小。如下图所示，三者具有相同的IOU，但是不能反映两个框是如何相交的，从直观上感觉第三种重合方式是最差的

![image-20240428151015756](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428151015756.png)



**GIOU的优势**

- -1 <= GIOU <= 1

- **GIOU关注重叠区域，也关注非重叠区域**
- **两个边框没有重叠的时候也可以计算损失**
- **0 <= lossGIOU <=2  GIOU loss的范围很小，网络不会剧烈波动，更有稳定性**

- ![image-20240428152422245](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428152422245.png)

- $A^{c}$是外接矩形的面积，u是 两个边框并集的面积，当两个框完全重合时， $A^{c}$ = u  所以分子为0，Giou退化成IOU

![image-20240428161033120](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428161033120.png)

![image-20240428152105664](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428152105664.png)

- GIOU加入了一个外接矩形



**GIOU损失**

- 0 <= lossGIOU <=2

![image-20240428153004628](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428153004628.png)









# DIOU

https://zhuanlan.zhihu.com/p/94799295

GIOU的缺点

- 当两个框在水平方向或垂直方向上时，GIOU会退化成IOU



DIOU

- -1 <= DIOU <= 1

- ![image-20240428153801590](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428153801590.png)

**DIOU的优点**

- **DIoU loss可以直接最小化两个目标框的距离，因此比GIoU loss收敛快得多。然后DIOU不会退化成IOU**
- **DIoU还可以替换普通的IoU评价策略，应用于NMS中，使得NMS得到的结果更加合理和有效。**



DIOU loss

- 0<= L DIOU <= 2

![image-20240428154142906](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428154142906.png)





# CIOU

**DIOU缺点**

- 边框的长宽比还没被考虑到计算中



**CIOU的优点**

- 加入了边框的长宽比





CIOU中  第二项为中心点距离，av是长宽比因素



![image-20240428162339261](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428162339261.png)



CIOU loss

![image-20240428162353982](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428162353982.png)