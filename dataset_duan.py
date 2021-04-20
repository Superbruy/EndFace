# -*- coding=utf-8 -*-
'''
#@ filename:  dataset_duan.py
#@ author:    Superbruy
#@ date:      2021-4-18
#@ brief:     dataset for end face
'''
import torch
from torch.utils.data import DataLoader, Dataset

import cv2 as cv
from PIL import Image

import os
import numpy as np
import random

#test
from utils_duan import ToTensor, Normalize
classes = ['good', 'bad']
SEED = 1234

class EndDataset(Dataset):
    def __init__(self, root, mode, split, transform=None):
        super(EndDataset, self).__init__()
        if mode not in ('train', 'test'):
            raise Exception("phase {} is not supported, choose train or test"\
                            .format(mode))
        self.mode = mode
        self.root = root
        self.split = split
        self.transform = transform

        self.data_info = self._get_image_info()
    def __getitem__(self, idx):
        image_path, label = self.data_info[idx]
        image = Image.open(image_path).convert('RGB')
        sample = {'image': image, 'label': label}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.data_info)

    def _get_image_info(self):
        img_path = []
        for dirname, foldername, filename in os.walk(self.root):
            # 分别是路径名，文件夹名，非文件夹名
            for name in filename:
                if name.endswith(".jpg"):
                    # print(dirname) ./duan/good
                    img_path.append(os.path.join(dirname, name))

        # os.path.basename: 返回路径最后的名字，若以/结尾，则返回空
        # os.path.dirname: 去掉最后的名字，返回前面的
        random.seed(SEED)
        random.shuffle(img_path)
        # print(img_path[0])  # ./duan/good/Image_20201028135352163.jpg

        img_label = [classes.index(os.path.basename(os.path.dirname(p))) for p in img_path]
        split_n = int(len(img_label) * self.split)
        if self.mode == 'train':
            img_set = img_path[:split_n]
            label_set = img_label[:split_n]
        else:
            img_set = img_path[split_n:]
            label_set = img_label[split_n:]
        assert len(img_set) == len(label_set)
        # print(img_set[0], label_set[0])
        # path_info = [os.path.join(self.root, p) for p in img_set]
        data_info = [(n, v) for (n, v) in zip(img_set, label_set)]
        # print(data_info[23])
        return data_info

if __name__ == '__main__':
    dataset = EndDataset('./duan', 'train', 0.9)
    dataset._get_image_info()
    # train_transform = transforms.Compose([
    #     GenerateDuan(),
    #     ToTensor(),
    #     Normalize(norm_mean, norm_std)
    # ])
    # train_dataloader = DataLoader(dataset, 8, True)
    # for i, sample in enumerate(train_dataloader):
    #     print(sample)