# -*- coding: utf-8 -*
import time
import numpy as np
import random
import os
from torch.nn import init
from Unet import UNet
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
import torch.backends.cudnn as cudnn
from ERA5 import ERA5_Dataset
from torch.utils.data import DataLoader
import sys

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#############################################################

'''
training Network and Val Network
'''
setup_seed(20)

def Train_Net(train_path,
              Iterator_epoch,
              Batchsize,
              save_path,
              val_path):
    '''
    :param train_path: 训练数据路径
    :param Iterator_epoch: 训练迭代次数
    :param Batchsize: 训练数据批次
    :param save_path: 模型保存路径
    :param val_path:  测试数据路径
    :return:
    '''

    UNet_model = UNet(3, 1)  # 输入数据的channel为3，输出数据的channel为1
    UNet_model.cuda()
    cudnn.benchmark = True
    UNet_model = nn.DataParallel(UNet_model)

    ####设置模型优化器以及损失函数loss_func
    optimizer = optim.Adam(UNet_model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_func = nn.MSELoss(reduction='sum')

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True,
                                               threshold=0.00001, threshold_mode="rel", cooldown=0, min_lr=0.0000001,
                                               eps=1e-08)

    ############下面部分需要根据自己的数据自行进行更改
    ####利用ERA5_Dataset加载训练数据集合

    train_Data = ERA5_Dataset(train_path, "Train", [48, 48], min_t2m=0, max_t2m=1)
    train_Data_Loader = DataLoader(train_Data, batch_size=Batchsize, shuffle=True, pin_memory=True, num_workers=4)

    ####利用ERA5_Dataset加载测试数据集合
    test_Data = ERA5_Dataset(val_path, "Test", [48, 48], min_t2m=0, max_t2m=1)
    test_Data_Loader = DataLoader(test_Data, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    train_loss = 0


    for epoch in range(Iterator_epoch):
        UNet_model.train()
        for i, (imgs, label) in enumerate(train_Data_Loader):

            imgs = imgs.cuda()
            label = label.cuda()
            img_pred = UNet_model(imgs)
            loss = loss_func(label, img_pred)

            ##优化器反向传播操作
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(i)
            train_loss += loss

            if i % 2000 == 0:
                print("epoch:{},iter:{},loss:{}".format(epoch, i, train_loss / 2000))
                train_loss = 0

        print(optimizer.state_dict()['param_groups'][0]['lr'])

        model_path = save_path + "model_{}.pkl".format(epoch)
        torch.save(UNet_model, model_path)
        val_loss = Val_Net(test_Data_Loader, model_path)
        scheduler.step(val_loss)

def Val_Net(loader, model_path):
    ###加载测试模型
    model = torch.load(model_path)
    model.cuda()
    model.eval()
    val_loss=0
    index = 0

    ###下面部分需要根据加载模型进行测试
    for img,label in loader:
        img = img.cuda()
        label = label.cuda()
        test_data = model(img)
        val_loss += np.square(label-test_data)
        index += 1
        label_size = label.shape[-1]*label.shape[-2]

    val_loss = np.sqrt(np.sum(val_loss)/label_size)

    return val_loss

if __name__ == "__main__":
    start = time.time()
    train_path = "/home/ium/Zyb/TC_Unet/dealwith_data/Train_path.npy"
    save_path = "/home/ium/Zyb/TC_Unet/checkpoint/model_2"
    val_path = "/home/ium/Zyb/TC_Unet/dealwith_data/Val_path.npy"
    test_path = "/home/ium/Zyb/TC_Unet/dealwith_data/Test_path.npy"

    Train_Net(train_path, 100, 32, save_path, test_path)
    # test_Data = ERA5_Dataset(val_path, "Test", [48,48], min_t2m=0, max_t2m=1)
    # test_Data_Loader = DataLoader(test_Data, batch_size=1, shuffle=False,pin_memory=True)
    # Val_Net(test_Data_Loader,"/home/ium/Zyb/TC_Unet/checkpoint/model_0.pkl")
    end = time.time()
    print(end - start)