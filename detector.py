import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


class Detector:

    def __init__(self):
        self.img_size = 640
        self.threshold = 0.5
        self.stride = 1

        self.weights = './weights/best-v6.1-0.987.pt'

        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)#加载权重文件
        model.to(self.device).eval()
        model.half()

        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]#可以在保持纵横比的前提下对图像做resize
        img = img[:, :, ::-1].transpose(2, 0, 1)#就相当于数学中的转置，在三维中就是三个轴之间的相互转换，对应着(z, y, x)轴
        img = np.ascontiguousarray(img)#函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        img = torch.from_numpy(img).to(self.device)#将numpy数组转换为PyTorch中的张量，返回的张量和img共享相同的内存。返回的张量不可调整大小
        img = img.half()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):

        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]#标注汽车的预测结果
        pred = pred.float()
        pred = non_max_suppression(pred, 0.3, 0.45, classes=0, agnostic=False)#进行非极大值抑制，对其中confidence较低的box和IOU较大的box进行过滤

        boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                #返回汽车框的坐标、名称和置信度
                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if lbl not in ['car']:
                        continue
                    if conf < self.threshold:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return boxes
