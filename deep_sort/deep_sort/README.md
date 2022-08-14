# Deep Sort 

This is the implemention of deep sort with pytorch.

ckpt.t7是一个特征提取网络的权重文件，特征提取网络训练好了以后会生成这个权重文件，方便在目标追踪的时候提取目标框中的特征，在目标追踪的时候避免ID switch。