import torch
from torchvision import transforms
from torch.utils.data import Sampler

import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import mrcfile

from code.refs import globalvar as gl
import torchio as tio
from torchio.transforms import (
    Compose,
)
import random

class_number = gl.get_value('class_number')

tio_random_scale = tio.RescaleIntensity(out_min_max=(0,1))
tio_random_flip = tio.RandomFlip(axes=(0, 1))

#train_loader = Compose([tio_random_flip])
#test_loader = Compose([tio_random_scale])
norm=Compose([tio_random_scale])
flip=Compose([tio_random_flip])
flipnorm=Compose([tio_random_flip,tio_random_scale])

unloader = transforms.ToPILImage()


def save_image(tensor, save_dir, name):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save(save_dir + '/' + name)

def save_mrc(tensor, save_dir, name):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    mrcimage = image.cpu().detach().numpy()
    mrcfile.new(save_dir + '/' + name, data=mrcimage.astype('float16'), overwrite=True)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])



def load_img(filepath):
    img = Image.open(filepath).convert('L')#.convert('YCbCr')
    #y, _, _ = img.split()
    return img


class DatasetFromFolder_norm(data.Dataset):
    def __init__(self, image_dir, target_dir,  input_transform=transforms.ToTensor(), target_transform=transforms.ToTensor()):
        super(DatasetFromFolder_norm, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.filenames = listdir(image_dir)
        ## !!!! 
        self.target_filenames = [join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = load_img(self.target_filenames[index])
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

    
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, target_dir,  input_transform=transforms.ToTensor(), target_transform=transforms.ToTensor()):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.filenames = listdir(image_dir)
        ## !!!! 
        self.target_filenames = [join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = load_img(self.target_filenames[index])
        if self.input_transform:
            input = self.input_transform(input)*255
        if self.target_transform:
            target = self.target_transform(target)*(class_number-1)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


class DatasetSampler(Sampler):
    def __init__(self, mask):
        self.mask = mask

    def __iter__(self):
        return (self.indices[i] for i in torch.nonzero(self.mask))

    def __len__(self):
        return len(self.mask)
'''
read .mrc file
'''
    
def is_mrc_file(filename):
    return any(filename.endswith(extension) for extension in [".mrc"])

def load_mrc(filepath):
    input_em = mrcfile.open(filepath, permissive=True);

    # print(np(input_em))
    #read mrc
    input_data= input_em.data
    input_data = np.array(input_data)
    #print('-------pytorch---type----tensor------')
    #img_tensor = torch.from_numpy(input_data)
    img_shape = np.shape(input_data)
#    print('image shape', img_shape)
    img_array = input_data.reshape((1, img_shape[0], img_shape[1], img_shape[2]))
    img_tensor = torch.from_numpy(img_array)
    #img_tensor = unloader(img_tensor)
 #   print(np.shape(img_array))
  #  print(type(img_array))
    img_array = np.array(img_array)
    return img_array
    
class DatasetFromFolder_mrc(data.Dataset):
    def __init__(self, image_dir, target_dir, input_transform=norm, target_transform=None):
        super(DatasetFromFolder_mrc, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_mrc_file(x)]
        self.filenames = listdir(image_dir)
        self.target_filenames = [join(target_dir, x) for x in listdir(target_dir) if is_mrc_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_mrc(self.image_filenames[index])
        target = load_mrc(self.target_filenames[index])
        #gen seed so that label and image will match when randomize
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


class DatasetFromFolder_mrc_train(data.Dataset):
    def __init__(self, image_dir, target_dir, input_transform=flipnorm, target_transform=flip):
        super(DatasetFromFolder_mrc_train, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_mrc_file(x)]
        self.filenames = listdir(image_dir)
        ## !!!! 
        self.target_filenames = [join(target_dir, x) for x in listdir(target_dir) if is_mrc_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_mrc(self.image_filenames[index])
        target = load_mrc(self.target_filenames[index])
        #gen seed so that label and image will match when randomize
        seed = np.random.randint(10000)
    
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # needed for torchvision 0.7
        #print('seed', seed)
        #print('first, ',torch.rand(1))

        if self.input_transform:
            input = self.input_transform(input)
        
        random.seed(seed) # apply this seed to target tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        torch.cuda.manual_seed_all(seed)
        #print('seed', seed)
        #print('second, ',torch.rand(1))
        
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolder_mrc_test(data.Dataset):
    def __init__(self, image_dir, target_dir, input_transform=norm, target_transform=None):
        super(DatasetFromFolder_mrc_test, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_mrc_file(x)]
        self.filenames = listdir(image_dir)
        ## !!!!
        self.target_filenames = [join(target_dir, x) for x in listdir(target_dir) if is_mrc_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_mrc(self.image_filenames[index])
        target = load_mrc(self.target_filenames[index])

        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
