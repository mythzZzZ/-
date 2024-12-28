

# Anchor base

## YOLOV1



- anchor：预定义两个anchor,v1中的anchor只是预测数值而已，没有正式很好的使用,所以说v1还是anchor-free

![image-20240422161646903](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240422161646903.png)

- backbone：darknet
  - 卷积，池化，softmax

- head 预测值

  - cls  20类别
  - reg 4
    - 位置信息,两个位置坐标相对于grid cell左上角,两个长宽相对于grid cell的比例
    - ![image-20240418213104698](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240418213104698.png)

  - confidence 1 (置信度代表了算法对于其预测结果的自信程度。简单地说，就是算法觉得“这个框里真的有一个物体”的概率)
    - confidence是算出来用来存放值得,不是预测出来的
    - 用来存放检测到的目标的分数,来进行极大值抑制
    - ![image-20240418211717478](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240418211717478.png)
    - 测试的confidence用来NMS极大值抑制,计算公式如下


  ![image-20240418211816527](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240418211816527.png)

- 损失函数

  - 类别损失，位置损失，有目标的confidence损失，没有目标的confidence损失



![image-20240418211900040](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240418211900040.png)

- 标签分配策略 (**bbox是网络预测后的bbox**)
  - 标签分配：GT的中心落在哪个grid，那个grid对应的两个bbox中与GT的IOU最大的bbox为正样本，其余为负样本，（由于是回归模型，不是分类模型，其解决类别不平衡的方式为各项loss采取不同的权重），即虽然一个grid分配两个bbox，但是只有一个bbox负责预测一个目标（边框和类别），这样导致YOLOv1最终只能预测7*7=49个目标。
    

- **缺点**
  - 预测的目标太少 7x7=49个目标
  - recall比较差（把图片上所有目标都检测出来）



## YOLOV2

- **Muti-Scale Training  多尺度训练**

- anchor:**使用聚类生成anchor**,reg得到的预测框的大小是相对于anchor生成的,使用5个anchor

https://zhuanlan.zhihu.com/p/432343631

![image-20240422160627327](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240422160627327.png)

- backbone：darknet,**加入BN**,只是用卷积和池化,没有FC

- head 13x13x5

  - cls
  - reg 4
    - ![image-20240418213251984](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240418213251984.png)

  - confidence (与v1一样)
    -  训练confidence
    - 测试confidence

- 损失函数：

  ![image-20240424204947254](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240424204947254.png)

  https://blog.csdn.net/just_sort/article/details/103232484?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171396295916800225552826%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171396295916800225552826&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-103232484-null-null.142

- 标签分配策略

  - 标签分配：（1）由YOLOv1的7 X 7个grid变为13 X 13个grid，划分的grid越多，多个目标中心落在一个grid的情况越少，越不容易漏检；（2）一个grid分配由训练集聚类得来的5个anchor（bbox）；（3）对于一个GT，首先确定其中心落在哪个grid，然后与该grid对应的5个bbox计算IOU，选择IOU最大的bbox负责该GT的预测，即该bbox为正样本；将每一个bbox与所有的GT计算IOU，若Max_IOU小于IOU阈值，则该bbox为负样本，其余的bbox忽略



## YOLOV3



- anchor x3

![image-20240422160753158](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240422160753158.png)

- backbone：darknet **抛弃了池化，使用卷积进行下采样**

- head (有三个头)

  - cls 80
  - reg
  - confidence
    - 训练confidence
    - 测试confidence

- **损失函数**：

  - ![image-20240418210408711](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240418210408711.png)
  - 置信度损失，二值交叉熵

  ![image-20240501153737421](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240501153737421.png)

  - 类别损失，二值交叉熵

![image-20240501154345675](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240501154345675.png)

   定位损失，均方误差

![image-20240501154759050](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240501154759050.png)





- **标签分配**

  ![image-20240424214332133](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240424214332133.png)

- **三个检测头**

  - 13x13x3 预测大物体
  - 26x26x3
  - 52x52x3 此时负责预测小物体

- 边界框预测 （与v2一样）

  ![image-20240418212556681](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240418212556681.png)

## YOLOv4

![image-20240422161921727](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240422161921727.png)

- CSPDarknet53
- **SPP的引入**
- **NECK**:**首次正式引入NECK结构** 
  - SPP 
  - PAN 在通道间进行concat

- Head：YOLOv3
- 位置预测：在yolov3中存在缺点
  - 因为$\sigma() = sigmod()$,sigmod只能将值映射到0-1所以 真实目标中心点非常靠近网络左上角点或者右下角点时，网络预测值需要达到正负无穷
  - 修改这个缺点作者引入缩放系数



![image-20240418214636636](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240418214636636.png)

- **正样本匹配策略**（IOUthreshold）
  
  - **从v4开始正样本匹配可以匹配多个anchor了，而且还可以选不同grid cell的anchor**
  - 通过GT与anchor计算IOU分配标签
  - 把GT的位置映射到grid cell上面，然后看GT的位置可以选哪几个cell的位置来匹配正样本，如下图可以选择三个cell来匹配正样本，要选择这三个cell的哪些anchor？通过计算GT与每一个anchor左上角重合时的交并比，当这个交并比大于某个阈值时，就选择该anchor，可以选择多个anchor，然后选中的三个cell都选择这些anchor为正样本
  - 引入了缩放因子，距离GT中心点(-0.5,1.5)范围内的anchor都能进行回归，但是GT的x，y举例左上角都小于0.5，所以上方grid cell 左上角起始位 + 1.几就能到达GT的位置（左边的gird cell也是），符合(-0.5,1.5)的范围，所以可以选择三个grid cell来分配正样本
  - ![image-20240418220242746](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240418220242746.png)
  
- 数据增强：mosaic

- 损失函数

  - 定位损失**CIOU**

  - ![image-20240418220857429](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240418220857429.png)

  - 置信度损失，二值交叉熵 和v3一样

    ![image-20240501153737421](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240501153737421.png)

    - 类别损失，二值交叉熵 和v3一样

  ![image-20240501154345675](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240501154345675.png)







## YOLOv5

- 各种数据增强
  - **mosaic**：四张图拼成一张

  - **copy paste**：将别的图的目标裁剪出来贴到当前图

  - **Random affine**：仿射变换，对拼接的图片进行平移和缩放

  - **MixUp**：将两张图片按照不同的透明度融合在一起

  - Albumentations:对图片滤波、直方图均匀化以及改变图片质量

  - Augment HSV：随机调整色度，饱和度 透明度

- 标签分配 一个GT可以分配给多个anchor

- backbone:CSP-Darknet53

- NECK:**SPPF**,New CSP-PAN
  - 传统SPP的特点：**并行池化**在concat

  - ![image-20240419092913696](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240419092913696.png)

  - **SPPF**：串行MaxPool

  - ![image-20240419093157814](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240419093157814.png)



- 使用focus
  - 把相邻的元素拼接在一起，相当于一个下采样，把元素都放到了channel维度
    - 这样做的好处，扩大了感受野，更容易检测到小目标


- 位置预测

![image-20240418215006141](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240418215006141.png)

- 激活函数：SILU



- 损失函数 （和yolov4一样）
  - Class loss: **bce loss(二值交叉熵损失)**
  - objectness loss:**bce loss(二值交叉熵损失)**
  - location loss  **CIOU**  

![image-20240419094721294](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240419094721294.png)

- 正样本匹配策略
  - **与yolov4相比，选取anchor的方式不一样，YOLOv5通过IOU的四倍和0.25倍来选取anchor(YOLOV4是>0.3)。其他都一样，cell的选取方式都一样**
  - ![image-20240419101020205](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240419101020205.png)





## SSD

YOLO算法难以检测小目标，SSD一定程度上克服了这个缺点



**backbone**

- YOLO只在最后一个特征层输出进行预测
- 空洞卷积

- SSD在多个特征层都进行检测

![image-20240429095601900](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429095601900.png)





**head**

- 六个头，不同的头先验框的尺寸不一样

![image-20240429101455879](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429101455879.png)



正负样本匹配

https://zhuanlan.zhihu.com/p/163600605

- 正样本
  - 对于图片中的每个gt_box，找到与其IOU最大的prior_box（这个特征图每一个grid cell都匹配），该先验框与其匹配，这样可以保证每个gt_box一定与某个prior_box匹配
  - 对于剩余未匹配的priors，若与某个gt*box的IOU大于某个阈值(一般0.5)，那么该priorbox与这个gt*_box匹配。
  - 这样可以先确定了第一个正样本，哪怕这个样本IOU很小（不超过0.5），但是也是有样本。然后看还有没有超过0.5的匹配GT，这样就可以分配到多个正样本
- 负样本
  - 为了解决负样本过多的问题，对负样本进行抽样，抽样时按照置信度误差（预测背景的置信度越小，误差越大）进行降序排列，选取误差的较大的top-k作为训练的负样本，以保证正负样本比例接近1:3







**损失函数**

- conf 置信度误差 （softmax）
- loc位置误差

![image-20240429103026436](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429103026436.png)



- 位置损失 smoothL1 是平方绝对值误差，减法误差的一种4
- ![image-20240429104411184](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429104411184.png)

- 置信度损失（类别预测在里面）

![image-20240429104437469](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429104437469.png)













![image-20240429101100346](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429101100346.png)



![image-20240429101133862](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429101133862.png)





## PP-YOLO

https://blog.csdn.net/qq_41375609/article/details/116375385

- backbone:ResNet50
  - 可变形卷积
  - 滑动平均策略





损失

- 位置损失最普通的IOU LOSS
- 交叉熵
- 交叉熵



Matrix NMS



## PP-YOLOv2

- Mish激活函数
- ![image-20240508212216032](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240508212216032.png)
- objness 使用分类 IOU 位置综合起来





























# two stage

## RCNN

https://zhuanlan.zhihu.com/p/23006190

https://zhuanlan.zhihu.com/p/52379393

https://www.bilibili.com/video/BV1af4y1m7iL/?spm_id_from=333.337.search-card.all.click&vd_source=101c15d8f637ac53427cd544709ff85d

VGG16

- 候选区域生成： 一张图像生成1K~2K个候选区域 （采用Selective Search 方法）（每生成一次要前向传播一次）
  - 使用一种过分割手段，将图像分割成小区域 (1k~2k 个)
  - 查看现有小区域，按照合并规则合并可能性最高的相邻两个区域。重复直到整张图像合并成一个区域位置
  - 对多个候选区域缩放到固定大小在给CNN

- 特征提取： 对每个候选区域，使用深度卷积网络提取特征 （CNN）
  - 由于文中使用的CNN中包含有全连接层，这就需要输入神经网络的图片有相同的size，但是Selective Search提取的Region Proposal都是不同size的，所以需要对每个Region Proposal都缩放到固定的大小（227*227）。paper试验了两种不同的处理方法
  - 通过CNN提取特征
  - 通过Selective Search挑选的候选框与GT计算IOU 正负样本（1：3）

![image-20240429202642765](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429202642765.png)

- 类别判断： 特征送入每一类的SVM 分类器，判别是否属于该类

  - SVM属于二分类，每一个类别都有一个SVM分类器

  - ![image-20240429202930291](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429202930291.png)

  - 2000 x 4096 （2000个框，每个框4096个特征） 4096 x 20（4096个SVM权重，20个SVM分类器），最后得到 2000 x 20 （每个框20个类别的概率）

  -  对 2000x20 计算IOU，进行同个类别下的NMS

![image-20240429203415016](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429203415016.png)



- 位置精修： 使用回归器精细修正候选框位置
  - NMS处理后剩余的候选框进一步筛选，用20个回归器对20个类别NMS筛选后剩下的候选框进行回归，最终得到每个类别修正后的得分最高的bounding box

![image-20240429155000116](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429155000116.png)



缺点

![image-20240429204215818](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429204215818.png)

损失函数

- 分类 交叉熵损失
- 回归 smooth L2 损失

## Fast RCNN

https://zhuanlan.zhihu.com/p/52379393

https://www.bilibili.com/video/BV1af4y1m7iL/?p=2&spm_id_from=pageDriver&vd_source=101c15d8f637ac53427cd544709ff85d

VGG16



- 一张图生成2k个候选区域（selective search方法）
- 将图像传递给卷积神经网络生成特征图，将ss生成的候选框投影到特征图上获得相应的**特征矩阵**；
  - 此时选择感兴趣的样本
- 每个特征矩阵通过ROI pooling 层缩到7x7大小的特征图，将特征图展平通过一系列全连接得到预测结果
  - softmax层用于全连接网以输出类别。与softmax层一起，也并行使用线性回归层，以输出预测类的边界框坐标。

![image-20240429210516602](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429210516602.png)



![image-20240429161923681](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429161923681.png)



损失函数

- softmax分类损失（交叉熵损失，对数损失）
  - ![image-20240501161352122](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240501161352122.png)

- smooth L1边界框回归损失 （RCNN是 smooth L2） （L1：最小绝对值偏差，L2：误差平方化，如果误差大于1 误差会放大很多）
  - ![image-20240501161445022](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240501161445022.png)

![image-20240429210621271](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429210621271.png)







## Faster RCNN

https://zhuanlan.zhihu.com/p/52379393

- 将图像作为输入并将其传递给卷积神经网络，返回该图像的特征图；
- 使用RPN结构生成候选框，将RPN生成的候选框投影到特征图上获得相应的特征矩阵
  - RPN在这些特征图上使用滑动窗口，滑动窗口每滑到一个位置后，生成一个一维的向量，**在通过两个全连接层分别输出得到类别概率和边界框回归参数**。类别概率是相对于预定义的anchor boxes生成的，假如预先定义k个anchor boxes，滑动窗口会生成2k个类别概率，每个anchor boxes对应2个类别概率值（是背景的概率，不是背景的概率）。k个anchor boxes会生成4k个reg，每个anchor boxes都会生成4个reg位置坐标
  - 滑动窗口滑完之后大约有6k个anchor，使用NMS后每张图片只剩2k个anchor
  - 最后使用256个anchor来组成正负样本（1：1 128个正样本 128个负样本）
    - 两种定义正样本方式
      - anchor与GT IOU超过0.7
      - anchor与GT IOU最大（最大不一定超过0.7，极少数条件下）
    - 负样本
      - anchor与GTIOU小于0.3
  - ![image-20240429212554348](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429212554348.png)

- 将每个特征矩阵通过ROI pooling层缩放到7x7大小的特征图，接着将特征图展平通过一系列全连接层得到预测结果





![image-20240429163811113](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429163811113.png)





损失函数

- ![image-20240429214314888](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429214314888.png)
- cls损失  交叉熵损失
  - ![image-20240429214756322](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429214756322.png)
- reg损失
  - smooth L1









# Anchor free



## FCOS



- **为什么提出FCOS?**

  - 作者认为目标检测的性能跟 anchor size有关
    - anchor base的anchor size都是固定的，所以很难处理形状变化的目标。如果迁移到其他任务中的话要重新设计anchor性能才会好

  - 检测器的性能跟正负样本是否均匀也有关

  - anchor base需要更多的参数



- backbone

![image-20240424193357462](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240424193357462.png)





- **head 输出**
- ![image-20240424194031788](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240424194031788.png)
  - 四个坐标 **（ltrb）**

![image-20240424193645263](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240424193645263.png)

- 80个类别
- 一个center-ness（用来分配正样本的）
  - center-ness用来表述距离目标中心点的远近程度，在0~1之间，距离目标中心点越近center-ness越接近与1。
  - ![image-20240424194244467](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240424194244467.png)
  - l,t,r,b是什么是grid cell 预测的中心点距离GT box左侧，上测，右侧，下侧的距离
  - **传统的confidence用  类别分数 x IOU，focs用类别分数 x center-ness来充当进行推理时的NMS得分**





- **正样本匹配策略**
  - anchor base是通过计算与anchor的IOU来匹配正样本，anchor free的fcos如何匹配呢？
  - **FCOS的正样本必须在GT的sub-box内**，sub-box如何计算,sub-box范围内所有点都是正样本
  - ![image-20240424195732098](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240424195732098.png)
  - r = 1.5  s为相对于原图的步距
  - ![image-20240424195802240](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240424195802240.png)







- **损失函数**
- ![image-20240424200301822](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240424200301822.png)
  - cls 二值交叉熵损失（log损失）
    - 采用bce_focal_loss,二值交叉熵损失配合focal_loss,计算损失时所有样本都参与计算（正样本与负样本）
  - Reg
    - giou_loss
  - center-ness
    - 二值交叉熵损失



- Ambiguity问题
  - 在匹配正样本时当特征图上的某一点同时落入多个GT Box内时，到底应该分配给哪一个GT的问题
    - 默认将该点分配给面积最小的GT Box
    - 但是该方法不能很好的解决问题，所以用了多个检测头，把目标划分到对应的特征图上
    - ![image-20240424202205346](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240424202205346.png)











## YOLOX

- **anchor free**
  - 预测的四个坐标 + objectness(IOU) 
  - objectness与上面yolo系列的confidence一样，为什么要预测confidence呢？在训练的时候可以通过GT来计算IOU，但是测试的时候就没有GT了，此时要预测confidence的值来进行NMS极大值过滤
  - anchor free 每一个cell只预测一个边框，与有anchor的不一样，所以分配标签的方式也有点不同

![image-20240419103956335](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240419103956335.png)





- **decoupled head**

![image-20240419103404188](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240419103404188.png)

- **avanced label assigning 正样本匹配策略** **SimOTA**
  - 在sub-box区域寻找正样本，如何寻找正样本呢？计算sub-box区域每一个cell的cost（cost由回归损失和类别损失组成）和IOU【如图2】，根据IOU从大到小排序，选取前k个IOU大的，然后构建GT与IOU的矩阵和GT与COST的矩阵【如图3】。通过IOU矩阵行求和得到向下取整的整数值，这个整数值就是对应GT要选取正样本的个数【如图4】。得到个数之后如何从矩阵选取正样本？**GT矩阵中每一行GT大的先选为正样本**【如图5】，如果两个GT有选中同样的正样本，则该样本只分配给cost更小的GT【如图6】

​                                                                         图1

![image-20240419111110032](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240419111110032.png)

​                                                            图2

![image-20240419111459368](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240419111459368.png)

​                                                            图3

![image-20240419112229863](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240419112229863.png)

​																图4



![image-20240419112435118](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240419112435118.png)

​																图5

![image-20240419112842205](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240419112842205.png)

​                                                          图6











- **损失函数**

  ![image-20240419104650162](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240419104650162.png)

类别损失

- 二值交叉熵

![image-20240501162728662](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240501162728662.png)



位置损失

- 这个就是IOULoss： -log(IOU)

![image-20240501162802949](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240501162802949.png)

置信度损失

- 二值交叉熵



## YOLOv6

![image-20240428102034106](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428102034106.png)

美团，工业

- 提出模型重参数化结构，训练时的特征模块与推理时的特征模块不一样
- **backbone**：RepVGGBlock

![image-20240422113103455](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240422113103455.png)



**head**

- 解耦头，cls输出类别信息（80），reg输出位置信息（5） **(ltrb)**   **objectness不进行loss计算**





**标签分配策略**

- TAL标签分配策略（Task alignment learning）

  - 传统的检测器的样本是基于IOU来分配的，分类最优的anchor和定位最优的anchor往往不是同一个，不能对两个任务同时做出准确的预测

  - 计算 $t=s^{a} \times u^{\beta}$  通过计算类别分数和预测框的IOU的高阶组合（s and u 分别为分类得分和 IoU 值，$\alphaα$ and β 分别为权重) https://blog.csdn.net/jiaoyangwm/article/details/119837303

  - 选择m个具有最大t值得anchor作为正样本点，其余的为负样本点 (这些样本都是在gt内)






- 损失函数
  - 分类损失：VFL
    - 提出了非对称的加权操作，针对正负样本有不平衡的问题和正样本中不等权的问题，来发现更多有价值的正样本。因此选择 VariFocal Loss 作为分类损失

  ![image-20240428112528045](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428112528045.png)

  - 位置损失：DFL （**概率损失**）t * GIOU loss 相当于也是均方误差
    - 将连续分布的box位置简化为离散的概率分布。它考虑了数据的模糊性和不确定性，而没有引入任何其他强的先验因素，这有助于提高box的定位精度，特别是当ground-truth boxes模糊时
    - ![image-20240428112446729](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428112446729.png)
  - **YOLOv6中没有使用objectness损失**

总的损失函数

- 使用了自蒸馏方案的损失函数

![image-20240428104057427](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428104057427.png)





## PP-YOLOE

**backbone**

- 使用了模型重参数化 **CSPRes中的主要特征提取模块是RepVggBlock**，网络中(backbone，neck，head)都使用ESE模块
- ![image-20240428113954561](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428113954561.png)

- ![image-20240428113414095](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428113414095.png)



**HEAD**

- 使用带ESE的解耦头，分别输出cls reg
  - cls 80
  - reg 5 （lrtb objectness（GIOU））



**样本分配策略** (https://zhuanlan.zhihu.com/p/505992733)

- TAL

![image-20240428120712839](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428120712839.png)



**损失函数**

- ![image-20240428120329166](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428120329166.png)







## YOLOv7

https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247504144&idx=1&sn=65c203e81a2c03225793a6d6f1ef7dea&scene=21#wechat_redirect

模型重参数话

- 模型重参化策略在推理阶段将多个模块合并为一个计算模块

Bag of freebies

- 在这里就是指用一些比较有用的训练技巧来训练模型，  只会改变训练策略或只会增加训练成本(不增加推理成本)的方法。从而使得模型获得更好的准确率但不增加模型的复杂度，也就不会增加推理的计算量。、



anchor base



**backbone**

![image-20240428204458317](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428204458317.png)

E-ELAN模块

- 利用分组卷积来扩展计算模块的通道和基数，将每个模块计算出的特征图根据设置的分组数打乱成G组，最后将它们连接在一起。

![image-20240428205933145](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428205933145.png)



**head**

- 引导头
- 辅助头

![image-20240428210303489](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428210303489.png)



**标签分配**

https://zhuanlan.zhihu.com/p/545768422

- **coarse-to-fine**（由粗到细)**引导标签分配策略**
  - lead_head 和 aux_head 分别选gird 位置
  - 使用simOTA分配标签



![image-20240428211518375](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428211518375.png)



**损失函数**

损失函数的值 == 目标置信度损失*0.1+类别置信度损失*0.125+坐标回归损失*0.05，在yolov7中的置信度损失和类别损失用的是二元交叉熵来做的，而定位损失是用的CIOU Loss来做的，跟yolov5是一样的







## YOLOv8







# Focal loss

https://blog.csdn.net/BIgHAo1/article/details/121783011

https://zhuanlan.zhihu.com/p/266023273  ***

YOLOv6和PP-YOLOE的损失函数 VFL DFL 都起源于Focal loss



![image-20240501171109675](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240501171109675.png)

- 即相比交叉熵损失，focal loss增加了一个因子（modulating factor），focal loss对于分类不准确的样本，损失没有改变，对于分类准确的样本，损失会变小。 整体而言，**相当于增加了分类不准确样本在损失函数中的权重**。
-   𝑝𝑡 因子也反应了分类的难易程度， 𝑝𝑡 越大，说明分类的置信度越高，代表样本越易分； 𝑝𝑡 越小，分类的置信度越低，代表样本越难分。因此**focal loss相当于增加了难分样本在损失函数的权重，使得损失函数倾向于难分的样本，有助于提高难分样本的准确度**。



![image-20240501171522910](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240501171522910.png)









# YOLO标签分配

https://blog.csdn.net/zhicai_liu/article/details/113631706





# NMS

https://zhuanlan.zhihu.com/p/78504109





https://zhuanlan.zhihu.com/p/54709759

![image-20240429103620060](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240429103620060.png)









# DETR



https://blog.csdn.net/weixin_43959709/article/details/115708159













# IOU



## IOU



https://blog.csdn.net/neil3611244/article/details/113794197

![image-20240428160504941](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428160504941.png)

![image-20240428150623109](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428150623109.png)



## GIOU

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









## DIOU

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





## CIOU

**DIOU缺点**

- 边框的长宽比还没被考虑到计算中



**CIOU的优点**

- 加入了边框的长宽比





CIOU中  第二项为中心点距离，av是长宽比因素



![image-20240428162339261](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428162339261.png)



CIOU loss

![image-20240428162353982](https://zhangwenkk333.oss-cn-beijing.aliyuncs.com/image/image-20240428162353982.png)
