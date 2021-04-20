# -*- coding=utf-8 -*-
'''
#@ filename:  train_tools.py
#@ author:    Superbruy
#@ date:      2021-4-19
#@ brief:     training API
'''
import numpy as np
import torch
import torch.nn as nn
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.models as models

class ModelTrainer:
    @staticmethod
    def train(data_loader, model, loss_fn, optimizer, schedular, epoch_id, device, max_epoch):
        conf_matrix = np.zeros((2, 2))  # total 2 classes
        loss_total = []
        acc = 0.
        for i, sample in enumerate(data_loader):
            input, label = sample['image'], sample['label']
            input, label = input.to(device), label.to(device)

            # forward
            output = model(input)

            # backward
            optimizer.zero_grad()
            loss = loss_fn(output, label)
            loss.backward()

            # update weights
            optimizer.step()

            # statistics
            pred = torch.argmax(output.data, dim=1)
            for j in range(len(label)):
                cate_i = label[j].cpu().numpy()
                pred_i = pred[j].cpu().numpy()
                conf_matrix[cate_i, pred_i] += 1

            loss_total.append(loss.item())
            acc = conf_matrix.trace() / conf_matrix.sum()

            schedular.step()

            if i % 10 == 0:
                print("Training Epoch[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] Loss{:.3f} Acc{:.3%}".format(
                    epoch_id, max_epoch, i+1, len(data_loader), np.mean(loss_total), acc
                ))
        return np.mean(loss_total), acc, conf_matrix

    @staticmethod
    def valid(data_loader, model, loss_fn, device):
        model.eval()

        conf_matrix = np.zeros((2, 2))
        loss_sigma = []
        with torch.no_grad():
            for i, sample in enumerate(data_loader):

                inputs, labels = sample['image'], sample['label']
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                # 统计预测信息
                _, predicted = torch.max(outputs.data, 1)

                # 统计混淆矩阵
                for j in range(len(labels)):
                    cate_i = labels[j].cpu().numpy()
                    pre_i = predicted[j].cpu().numpy()
                    conf_matrix[cate_i, pre_i] += 1.

                # 统计loss
                loss_sigma.append(loss.item())

        acc_avg = conf_matrix.trace() / conf_matrix.sum()

        return np.mean(loss_sigma), acc_avg, conf_matrix


def show_confMat(confusion_mat, classes, set_name, out_dir, verbose=False):
    """
    混淆矩阵绘制
    :param confusion_mat:
    :param classes: 类别名
    :param set_name: trian/valid
    :param out_dir:
    :return:
    """
    cls_num = len(classes)
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 显示

    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix' + set_name + '.png'))
    # plt.show()
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))))


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()