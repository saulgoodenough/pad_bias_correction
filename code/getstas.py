import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import time
import os
from PIL import Image
from PIL import ImageFile
#解决图像读取问题
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as pyplot
import numpy as np
import logging
import logging.config
#from metric_learning import ArcMarginProduct, AddMarginProduct, AdaCos
import pretrainedmodels
from tqdm import tqdm



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 均已测试


transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomCrop(224, padding=0),  #先四周填充0，再把图像随机裁剪成224*224
        transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomAffine(45),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
trainset = torchvision.datasets.ImageFolder(root='../train_data', transform=transform_train)

means = []
stds = []
for img in tqdm(trainset):
    means.append([torch.mean(img[0][0]).data, torch.mean(img[0][1]).data, torch.mean(img[0][2]).data])
    stds.append([torch.std(img[0][0]).data, torch.std(img[0][1]).data, torch.std(img[0][2]).data])

mean = torch.mean(torch.tensor(means), axis = 0).data.numpy()
print('mean', mean)
std = torch.mean(torch.tensor(stds), axis = 0).data.numpy()
print('std', std)

'''
mean [0.42574355 0.4047814  0.39227948]
std [0.28109735 0.27269357 0.26959908]
'''