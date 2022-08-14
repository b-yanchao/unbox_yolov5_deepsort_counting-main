[deepsort文件功能详解](https://blog.csdn.net/didiaopao/article/details/120274519?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-13-120274519.pc_agg_new_rank&utm_term=%E7%9B%AE%E6%A0%87%E8%BF%BD%E8%B8%AAdeepsort&spm=1000.2123.3001.4430)

**1 Configs文件目录下：**

   deep_sort.yaml：这个yaml文件主要是保存一些参数。

（1）里面有特征提取权重的目录路径；

（2）最大余弦距离，用于级联匹配，如果大于该阈值，则忽略。

（3）检测结果置信度阈值

（4）非极大抑制阈值，设置为1代表不进行抑制

（5）最大IOU阈值

（6）最大寿命，也就是经过MAX_AGE帧没有追踪到该物体，就将该轨迹变为删除态。

（7）最高击中次数，如果击中该次数，就由不确定态转为确定态。

（8）最大保存特征帧数，如果超过该帧数，将进行滚动保存。
**2 deep_sort/deep_sort/deep目录下**

ckpt.t7：这是一个特征提取网络的权重文件，特征提取网络训练好了以后会生成这个权重文件，方便在目标追踪的时候提取目标框中的特征，在目标追踪的时候避免ID switch。
evaluate.py：计算特征提取模型精确度。

feature_extractor.py：提取对应bounding box中的特征, 得到一个固定维度的特征，作为该bounding box的代表，供计算相似度时使用。

model.py：特征提取网络模型，该模型用来提取训练特征提取网络权重。[训练特征提取权重](https://blog.csdn.net/didiaopao/article/details/120276922?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164794186616780261959233%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=164794186616780261959233&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-8-120276922.142^v3^pc_search_result_cache,143^v4^control&utm_term=%E7%82%AE%E5%93%A5%E5%B8%A6%E4%BD%A0%E5%AD%A6&spm=1018.2226.3001.4187)

original_model.py：特征提取网络模型，可代替model。

train.py：训练特征提取网络的python文件

test.py：测试训练好的特征提取网络的性能
**3 deep_sort/deep_sort/sort目录下：**

detection.py：保存通过目标检测的一个检测框框，以及该框的置信度和获取的特征；同时还提供了框框的各种格式的转化方法。

iou_matching.py：计算两个框框之间的IOU。

kalman_filter.py：卡尔曼滤波器的相关代码，主要是利用卡尔曼滤波来预测检测框的轨迹信息。

linear_assignment.py：利用匈牙利算法匹配预测的轨迹框和检测框最佳匹配效果。

nn_matching.py：通过计算欧氏距离、余弦距离等距离来计算最近领距离。

preprocessing.py：非极大抑制代码，利用非极大抑制算法将最优的检测框输出。

track.py：主要储存的是轨迹信息，其中包括轨迹框的位置和速度信息，轨迹框的ID和状态，其中状态包括三种，一种是确定态、不确定态、删除态三种状态。

tracker.py：保存了所有的轨迹信息，负责初始化第一帧，卡尔曼滤波的预测和更新，负责级联匹配,IOU匹配。

deep_sort/deep_sort/deep_sort.py：deepsort的整体封装，实现一个deepsort追踪的一个整体效果。

deep_sort/utils：这里最主要有一些各种各样的工具python代码，例如画框工具，日志保存工具等等。

detector.py：封装的一个目标检测器，对视频中的物体进行检测

tracker.py：封装了一个目标追踪器，对检测的物体进行追踪
