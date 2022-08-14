'''
加载EfficientNet权重模型
'''
import logging

import cv2
from torchvision.transforms import transforms
import numpy as np
from deep_sort.deep_sort.deep.resnet_ibn_a import resnet50_ibn_a, resnet50_ibn_a_cls

import torch

class CLS(object):
    def __init__(self, model_path='', use_cuda=True):
        self.net = torch.load('deep_sort/deep_sort/deep/checkpoint/0.998-86.pt')
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        # state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        # self.net.load_state_dict(state_dict)
        # self.net.load_param(model_path)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.net.eval()
        self.size = (224, 224)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            outputs = self.net(im_batch)
            _, predicted = torch.max(outputs, 1)
        return predicted
