<<<<<<< HEAD
# yolov5 deepsort 行人 车辆 跟踪 检测 计数

- 实现了 出/入 分别计数。
- 默认是 南/北 方向检测，若要检测不同位置和方向，可在 main.py 文件第13行和21行，修改2个polygon的点。
- 默认检测类别：小客车、大客车、小货车、中货车、大货车、集装箱车。
- 检测类别可在 detector.py 文件第60行修改。


## 运行环境

- python 3.6+，pip 20+
- pytorch
- pip install -r requirements.txt


## 如何运行

2. 进入目录

    ```
    $ cd unbox_yolov5_deepsort_counting
    ```

3. 创建 python 虚拟环境

    ```
    $ python3 -m venv venv
    ```

4. 激活虚拟环境

    ```
    $ source venv/bin/activate
    ```
   
5. 升级pip

    ```
    $ python -m pip install --upgrade pip
    ```

6. 安装pytorch

    > 根据你的操作系统、安装工具以及CUDA版本，在 https://pytorch.org/get-started/locally/ 找到对应的安装命令。我的环境是 ubuntu 18.04.5、pip、CUDA 11.0。

    ```
    $ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    ```
   
7. 安装软件包

    ```
    $ pip install -r requirements.txt
    ```

8. 在 main.py 文件中第66行，设置要检测的视频文件路径，默认为 './video/test.mp4'

    > 140MB的测试视频可以在这里下载：https://pan.baidu.com/s/1geqjht-no0iyzQ88JQopwA  密码: i6cs

    ```
    capture = cv2.VideoCapture('./video/test.mp4')
    ```
   
9. 运行程序

    ```
    python main.py
    ```


## 引用

- https://github.com/Sharpiless/Yolov5-deepsort-inference
- https://github.com/ultralytics/yolov5/
- https://github.com/ZQPei/deep_sort_pytorch
=======
# unbox_yolov5_deepsort_counting-main

#### 介绍
{**以下是 Gitee 平台说明，您可以替换此简介**
Gitee 是 OSCHINA 推出的基于 Git 的代码托管平台（同时支持 SVN）。专为开发者提供稳定、高效、安全的云端软件开发协作平台
无论是个人、团队、或是企业，都能够用 Gitee 实现代码托管、项目管理、协作开发。企业项目请看 [https://gitee.com/enterprises](https://gitee.com/enterprises)}

#### 软件架构
软件架构说明


#### 安装教程

1.  xxxx
2.  xxxx
3.  xxxx

#### 使用说明

1.  xxxx
2.  xxxx
3.  xxxx

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
>>>>>>> 8f898b79e21d6b55798309733e21f56cdb62ceac
