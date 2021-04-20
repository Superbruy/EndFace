# -*- coding=utf-8 -*-
'''
#@ filename:  train_duan.py
#@ author:    Superbruy
#@ date:      2021-4-18
#@ brief:     train end face
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms

import os
import numpy as np
from datetime import datetime

from dataset_duan import EndDataset
from utils_duan import CosineWarmUp, GenerateDuan, ToTensor, Normalize
from train_tools import ModelTrainer, show_confMat, plot_line

Base_Dir = os.path.abspath(os.path.dirname(__file__))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    # config
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 1e-4

    batchsize = 8
    max_epoch = 20
    base_lr = 0.1
    final_lr = 1e-5
    warmup_epoch = 2
    printf_interval = 5

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    classes_names = ('good', 'bad')

    model_name = 'vgg19-dcbb9e9d.pth'
    model_dir = os.path.join(Base_Dir, 'data', model_name)
    dataset_root = os.path.join(Base_Dir, 'duan')
    log_dir = os.path.join(Base_Dir, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # ============================ step 1/5 data ============================

    train_transform = transforms.Compose([
        GenerateDuan(),
        ToTensor(),
        Normalize(norm_mean, norm_std)
    ])
    train_dataset = EndDataset(dataset_root, 'train', 0.9, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batchsize, True)

    test_transform = transforms.Compose([
        GenerateDuan(),
        ToTensor(),
        Normalize(norm_mean, norm_std)
    ])
    test_dataset = EndDataset(dataset_root, 'test', 0.9, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batchsize, True)

    batches = len(train_dataset) / batchsize  # iterations per epoch

    # ============================ step 2/5 model ============================
    model = models.vgg19(pretrained=False)
    # print(model)
    state = torch.load(model_dir)
    model.load_state_dict(state)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5, False),
        nn.Linear(4096, 4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5, False),
        nn.Linear(4096, 2, bias=True)
    )
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 2)
    model = model.to(device)

    # ============================ step 3/5 loss function ============================
    loss_fn = nn.CrossEntropyLoss()

    # ============================ step 4/5 optimizer ============================
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                          weight_decay=weight_decay)
    schedular = CosineWarmUp(optimizer, batches=batches, max_epoch=max_epoch, base_lr=base_lr,
                             final_lr=final_lr, warmup_epoch=warmup_epoch, warmup_init_lr=0)

    # ============================ step 5/5 train ============================
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0

    for epoch in range(max_epoch):
        loss_train, acc_train, mat_train = ModelTrainer.train(
            train_dataloader, model, loss_fn, optimizer, schedular, epoch, device, max_epoch)
        loss_test, acc_test, mat_test = ModelTrainer.valid(
            test_dataloader, model, loss_fn, device
        )
        print(
            "Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Test loss:{:.4f} LR:{}".format(
                epoch + 1, max_epoch, acc_train, acc_test, loss_train, loss_test, optimizer.param_groups[0]["lr"]))
        # 绘图
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_test)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(loss_test)

        show_confMat(mat_train, classes_names, "train", log_dir, verbose=epoch == max_epoch - 1)
        show_confMat(mat_test, classes_names, "valid", log_dir, verbose=epoch == max_epoch - 1)

        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        if epoch > (max_epoch / 2) and best_acc < acc_test:
            best_acc = acc_test
            best_epoch = epoch

            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}

            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)
    print(" done ~~~~ {}, best acc: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                                                               best_acc, best_epoch))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)