# EfficientNet模型

## 改进思路

通过同时探索输入图像分辨率、网络深度和宽度的影响，得出最佳组合。

![image-20220325161727336](C:\Users\21163\AppData\Roaming\Typora\typora-user-images\image-20220325161727336.png)



(a)传统的卷积神经网络

(b)在传统的卷积神经网络的基础上增加了宽度，即增加了每个卷积层的卷积核的个数，提升特征矩阵的channel的个数。

(c)在传统的卷积神经网络的基础上增加了网络的深度，layer个数增加。

(d)在传统的卷积神经网络的基础上增加了输入图像的分辨率，提升分辨率会导致得到的每个特征矩阵的高和宽都会增加。

(e)就是EfficientNet模型中讨论的，在在传统的卷积神经网络的基础上同时增加网络的宽度和深度以及输入图像分辨率。



## 三种改进对模型的影响

1、增加网络的深度depth能够得到更加丰富、复杂的特征并且能够很好地应用到其他它任务中。但网络的深度过深会面临梯度消失，训练困难的问题。

2、增加网络的width能够获得更高粒度的特征并且也更容易训练，但对于width很大而深度较浅的网络往往很难学习到更深层次的特征。

3、增加输入网络的图像分辨率能够潜在得获得更高细粒度的特征模板，但对于非常高的输入分辨率，准确率的增益也会减少。并且大分辨率图像会增加计算量。

## 模型效果

提出的EfficientNet-B7在Imagenet top-1上达到了当年的最高准确率84.4%，与其之前准确率最高的GPipe相比，参数量为其1/8.4，推理速度提升了6.1倍。

![image-20220325163023492](C:\Users\21163\AppData\Roaming\Typora\typora-user-images\image-20220325163023492.png)

****![image-20220325165612088](C:\Users\21163\AppData\Roaming\Typora\typora-user-images\image-20220325165612088.png)

## **EfficientNet**

![image-20220325170442670](C:\Users\21163\AppData\Roaming\Typora\typora-user-images\image-20220325170442670.png)

表中卷积层后默认都跟有BN以及Swish激活函数

Operator:操作

Resolution:输入图像分辨率

Channels：每个输出特征矩阵的Channel（通道）个数，也是卷积核个数，通过增加卷积核个数，增加网络宽度。

Layers:Operator重复次数，增加深度

stride：步长，如1输入224X224输出112X112，故步长为2

### MBConv



![image-20220325210944792](C:\Users\21163\AppData\Roaming\Typora\typora-user-images\image-20220325210944792.png)

1、第一个升维的1X1卷积层，他的卷积核个数是输入特征矩阵channel的n倍，n为MBConv1后的1或MBConv6后的6。

2、当n = 1时，不要第一个升维的1X1卷积层，即Stage2中的MBConv结构都没有第一个升维的1X1卷积层。

3、关于捷径分支(shortcut)连接，仅当输入MBConv结构的特征矩阵与输出的特征矩阵shape相同时才存在。

这里的Depwise Conv卷积的卷积核可能是3X3或5X5，具体参数看上表

### SE模块（注意力机制）

![image-20220325211211451](C:\Users\21163\AppData\Roaming\Typora\typora-user-images\image-20220325211211451.png)

由一个全局平局池化，两个全连接层。第一个全连接层的节点个数是输入该MBConv特征矩阵channels的1/4，且使用Swish激活函数。第二个全连接层的节点个数等于Depthwise Conv层输出的特征矩阵channels，且使用Sigmoid激活函数。

![image-20220325212314983](C:\Users\21163\AppData\Roaming\Typora\typora-user-images\image-20220325212314983.png)



width_coefficient代表channel维度上的倍率因子，比如在EfficientNet-b0中Stage1的3X3卷积层所使用的卷积核个数是32，那么在b6中就是32X1.8 = 57.6 接着取整到离它最近的8的整数倍即56。

depth_coefficient代表depth维度上的倍率因子（仅针对Stage2到Stage8），比如在EfficientNet-b0中Stage7的L = 4，那么在b6中就是4 X 2.6 = 10.4，接着向上取整即11。

drop_connect_rate:为MBConv中Dropout的随机失活比例，并不是固定的0.2，而是从0逐渐增长到0.2。

dropout_rate:为Stage9中的一个dropout层的随机失活比例。



代码实例：

https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test9_efficientNet





