'''
以2m温度为示例进行数据读取
'''
import numpy as np
import random
import time
import torch
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

transforms = transforms.Compose([transforms.ToTensor()])


class ERA5_Dataset(Dataset):

    def __init__(self, Data_path, State, Size, min_t2m, max_t2m, Transforms=None,
                 geo_path="/home/ium/Zyb/geo.npy"):
        '''
        :param Data_path:数据路径
        :param State: 表示当前读取训练数据还是测试数据
        :param Size: 表示需要截取的数据打大小，此处裁剪子图大小为（48*48)
        :param min_t2m: 2m温度最大值
        :param max_t2m: 2m温度最小值
        :param Transforms: 是否进行归一化
        :param geo_path: 地形数据的路径，与输入数据一起加入进行训练
        '''

        ####进行训练数据的shuffle操作，保证训练数据的随机性
        self.Paths = natsorted(np.load(Data_path))

        #加载地形数据并进行归一化
        self.geo = np.load(geo_path)
        self.geo = (self.geo - np.min(self.geo)) / (np.max(self.geo) - np.min(self.geo))

        #增加训练数据数据集的数目
        if State == "Train":
            self.Total = len(self.Paths) * 30
        else:
            self.Total = len(self.Paths)

        self.ele = "t2m"
        self.Data_Size = len(self.Paths)
        self.min_t2m = min_t2m
        self.max_t2m = max_t2m
        self.Transforms = Transforms
        self.State = State
        self.Global_Shape = [96, 192]
        self.Size = Size

        self.Data = self.load_2Ram(self.Paths)

    #加载数据由于Paths中保存的是文件的路径因此需要根据路径读取数据
    def load_2Ram(self, Paths):

        Data_Ram = {"paths": [],
                    "input": [],
                    "label": [],
                    "Ana": []}

        start = time.time()

        for path in Paths:

            data = np.load(path)
            if self.ele == "t2m":

                Data_Ram["paths"].append(path)
                Data_Ram["input"].append(data["For_{}".format(self.ele)] - 273.15)
                Data_Ram["label"].append(data["Label_{}".format(self.ele)] - 273.15)
                Data_Ram["Ana"].append(data["Ana_{}".format(self.ele)] - 273.15)
            else:
                Data_Ram["paths"].append(path)
                Data_Ram["input"].append(data["For_{}".format(self.ele)])
                Data_Ram["label"].append(data["Label_{}".format(self.ele)])
                Data_Ram["Ana"].append(data["Ana_{}".format(self.ele)])

        end = time.time()
        print("test_load_used:{}".format(end - start))
        return Data_Ram

    #类似于迭代器的功能向网络迭代提供数据
    def __getitem__(self, index):
        '''
        :param index: 表示读取数据的编号，训练集最大编号为 len(self.Paths) * 30
                                        测试集最大编号为 len(self.Paths)
        :return:
        '''

        img = None
        label = None

        #根据x_index以及y_index来实现随机裁剪
        x_index = np.random.randint(0, self.Global_Shape[0] - self.Size[0])
        y_index = np.random.randint(0, self.Global_Shape[1] - self.Size[1])
        # print(x_index,y_index)

        if self.State == "Train":
            img = self.Data["input"][index % self.Data_Size][np.newaxis, x_index:x_index + self.Size[0],
                  y_index:y_index + self.Size[1]]
            img_Ana = self.Data["Ana"][index % self.Data_Size][np.newaxis, x_index:x_index + self.Size[0],
                   y_index:y_index + self.Size[1]]
            label = self.Data["label"][index % self.Data_Size][np.newaxis, x_index:x_index + self.Size[0],
                    y_index:y_index + self.Size[1]]
            geo = self.geo[np.newaxis, x_index:x_index + self.Size[0], y_index:y_index + self.Size[1]]
            img = np.concatenate((img, img_Ana, geo), axis=0)
            # geo = torch.from_numpy(geo).float()

            img = torch.from_numpy(img).float()
            label = torch.from_numpy(label).float()
            # file_name = self.Data["paths"][index%self.Data_Size].split("/")[-1][:-4]
            # print(self.Data["paths"][index%self.Data_Size])
            return img, label

        else:
            img = self.Data["input"][index][np.newaxis, :, :]
            label = self.Data["label"][index][np.newaxis, :, :]
            img_Ana = self.Data["Ana"][index][np.newaxis, :, :]
            file_name = self.Data["paths"][index % self.Data_Size].split("/")[-1][:-4]
            # geo_x,geo_y = [int(x) for x in file_name.split("_")[1:]]
            # print(geo_x,geo_y)
            # geo = self.geo[np.newaxis,geo_x:geo_x+self.Size[0],geo_y:geo_y+self.Size[1]]
            img = np.concatenate((img[:, :96, :192], img_Ana[:, :96, :192]), axis=0)

            # geo = self.geo[np.newaxis,0:10,0:10]
            # geo = self.Data["paths"][index]

            #img表示网络最终输入数据，label表示网络学习的标签，即真实观测值
            img = torch.from_numpy(img).float()
            label = torch.from_numpy(label).float()
            # print(img.shape)
            return img, label

    def __len__(self):
        return self.Total


if __name__ == "__main__":

    train_Data = ERA5_Dataset("/home/ium/Zyb/TC_Unet/dealwith_data/Train_path.npy", "Test", [48, 48], min_t2m=0,
                              max_t2m=1)
    train_Data_Loader = DataLoader(train_Data, batch_size=1, shuffle=False, pin_memory=True)
    min_data = 1000
    max_data = 0

    for i, (imgs, label) in enumerate(train_Data_Loader):

        if (imgs.max() > max_data):
            max_data = imgs.max()
        if (label.max() > max_data):
            max_data = label.max()

        if (imgs.min() < min_data):
            min_data = imgs.min()
        if (label.min() < min_data):
            min_data = label.min()

        print(i)

    print(max_data)
    print(min_data)