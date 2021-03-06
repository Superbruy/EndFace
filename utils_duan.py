# -*- coding=utf-8 -*-
__author__ = 'Superbruy'

import torch
from torch.utils.data import WeightedRandomSampler
import torch.optim
from torch.optim.optimizer import Optimizer

from math import pi, cos
import cv2 as cv
import numpy as np
from PIL import Image

import os

class CosineWarmUp:
    '''
    :brief this warmup method consists of two parts, namely, linear warmup strategy and cosine lr decay
    :arg
    warmup stage:lr self.warmup_init_lr --> self.base_lr
    cosine stage:lr self.base_lr --> self.target_lr
    '''
    def __init__(self, optimizer, batches, max_epoch, base_lr, final_lr=0,
                 warmup_epoch=0, warmup_init_lr=0, last_iter=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('expect to get a Optimizer type but got a {}'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_iter = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lr = base_lr
        self.learning_rate = base_lr
        self.total_iters = max_epoch * batches
        self.target_lr = final_lr
        self.warmup_iters = warmup_epoch * batches
        self.warmup_init_lr = warmup_init_lr
        self.last_iter = last_iter
        self.step()

    def get_lr(self):
        if self.last_iter < self.warmup_iters:
            self.learning_rate = self.warmup_init_lr + \
                                 (self.base_lr - self.warmup_init_lr) / self.warmup_iters * self.last_iter

        else:

            self.learning_rate = self.target_lr + (self.base_lr - self.target_lr) * \
                                 (1 + cos(pi * (self.last_iter - self.warmup_iters) /
                                          (self.total_iters - self.warmup_iters))) / 2

    def step(self, iteration=None):
        """Update status of lr.
        Args:
            iteration(int, optional): now training iteration of all max_epochs.
                Normally need not to set it manually.
        """
        if iteration is None:
            iteration = self.last_iter + 1
        self.last_iter = iteration
        self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate


def _is_numpy_image(img):
    return img.ndim in {2, 3}


def get_c_r_naive_duan(image):
    img = np.asarray(image)
    img_shape = img.shape
    # convert bgr format to gray
    if len(img_shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(img, (9, 9), 0)
    (_, thresh) = cv.threshold(blurred, 90, 255, cv.THRESH_BINARY)

    # ?????????
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    # ????????????
    closed = cv.erode(closed, None, iterations=1)
    closed = cv.dilate(closed, None, iterations=4)

    # ????????????
    (_, cnts, _) = cv.findContours(
        # ???????????? ???????????????
        closed.copy(),
        # ????????????????????????
        cv.RETR_EXTERNAL,  # ????????????????????????
        # cv2.RETR_CCOMP,                #???????????????????????????,??????????????????
        # cv2.RETR_LIST,                 #????????????????????????????????????
        # cv2.RETR_TREE,                 #????????????????????????????????????
        # cv2.CHAIN_APPROX_NONE,         #????????????????????????????????????????????????????????????????????????1
        # ??????????????????????????????
        cv.CHAIN_APPROX_SIMPLE,  # ??????????????????????????????4???????????????????????????
        # cv2.CHAIN_APPROX_TC89_L1,
        # cv2.CHAIN_APPROX_TC89_KCOS
    )

    center = []
    radius = []
    for i in cnts:
        x, y, w, h = cv.boundingRect(i)

        if 1030 <= h <= 1130 and 1030 <= w <= 1130:
            # print(w, h, x, y)
            c = (int(x + w / 2), int(y + h / 2))
            r = int(w // 2)
            center.append(c)
            radius.append(r)
    print(center, radius)
    return center, radius

class GenerateDuan:
    '''
    input image: PIL format
    '''
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        w, h = image.size
        center, radius = get_c_r_naive_duan(image)
        center, radius = center[0], radius[0]

        #  in case the boundary is exceeded
        if radius > center[1]:
            left_up_corner = (center[0] - radius, 0)
        else:
            left_up_corner = (center[0] - radius, center[1] - radius)
        if center[1] + radius > h:
            right_bottom_corner = (center[0] + radius, h)
        else:
            right_bottom_corner = (center[0] + radius, center[1] + radius)
        image = image.crop((left_up_corner[0], left_up_corner[1], right_bottom_corner[0], right_bottom_corner[1]))

        image = image.resize((512, 512))

        return {'image': image, 'label': label}

class MixUp:
    '''
    each bad sample has a probability to mix up
    '''
    def __init__(self, p, good_root):
        self.p = p
        self.root = good_root
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # good sample do not need to mix up
        if label == 0:
            return {'image': image, 'label': label}

        # load good image randomly
        good_list = os.listdir(self.root)
        idx = np.random.choice(good_list)
        good_img_path = os.path.join(self.root, idx)
        good_sample = cv.imread(good_img_path)

        # mix up
        lam = np.random.beta(1., 1.)
        bad_sample = np.array(image)
        mix_image = (good_sample*lam + bad_sample*(1-lam)).astype(np.uint8)
        mix_image = cv.cvtColor(mix_image, cv.COLOR_BGR2RGB)
        mix_image = Image.fromarray(mix_image).convert('RGB')

        label = 1-lam
        return {'image': mix_image, 'label': label}

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.array(image)
        if not _is_numpy_image(image):
            raise Exception('expect dim to be 2 or 3, got {} instead'.format(image.ndim))
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
                image = np.repeat(image, 3, axis=2)
            image = torch.from_numpy(image.transpose((2, 0, 1)))
            if isinstance(image, torch.ByteTensor):
                return {'image': image.float().div(255), 'label': int(label)}
            else:
                return {'image': image, 'label': int(label)}

def _Normalize(tensor, mean, std, inplace=False):
    '''
    simulate the original function of normalization
    '''
    if not inplace:
        tensor = tensor.clone()
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)  # ????????????
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor

class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        img_tensor, label = sample['image'], sample['label']
        img_tensor = _Normalize(img_tensor, self.mean, self.std, self.inplace)

        return {'image': img_tensor, 'label': label}