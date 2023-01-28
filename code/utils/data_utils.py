import torch
from torchvision import transforms
from torch.utils.data import Sampler

import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
import numpy as np
'''
import torchio as tio
from torchio.transforms import (
    Compose,
)
'''
import random

## MRI related packages
import nibabel as nib

## age label utils
import sys
sys.path.append('../')

from dp_model import dp_utils as dpu

import glob

from torch.nn import Module

from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.decomposition import IncrementalPCA

from sklearn.preprocessing import StandardScaler

from scipy import sparse as sp

import pickle

import skimage.measure



def count_parameters(module: Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

#tio_random_scale = tio.RescaleIntensity(out_min_max=(0,1))
#norm=Compose([tio_random_scale])

def loadmri(mri_path):
    image_obj = nib.load(mri_path)
    image_data = image_obj.get_fdata()
    return image_data


def make_weights_for_balanced_classes(dataset, nclasses=48, minus_age = 38):
    count = [0] * nclasses
    for item in dataset:
        count[int(item[4][0]- minus_age)] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(dataset)
    for idx, val in enumerate(dataset):
        weight[idx] = weight_per_class[val[1]]
    return weight



class DatasetFromFolder_mri(data.Dataset):
    def __init__(self, cfg,  image_dir = None, target_dict_path = None, filelist = None, input_transform=None, target_transform=None):
        super(DatasetFromFolder_mri, self).__init__()
        self.cfg = cfg
        self.target_dict = np.load(target_dict_path, allow_pickle=True).item()
        self.bin_range = self.cfg.NET_DATALOADER.BIN_RANGE
        self.input_size = self.cfg.NET_INPUT.CROP_SIZE
        self.bin_step = self.cfg.NET_DATALOADER.BIN_STEP
        self.sigma = self.cfg.NET_DATALOADER.SIGMA

        if filelist is not None:
            self.image_filenames = filelist
        else:
            self.endlist = self.cfg.NET_DATALOADER.ENDLIST
            self.image_dir = image_dir

            self.image_filenames = self.get_all_file()

        self.image_filenames.remove('/home2/migrate_handan/Biobank_images/fmri_2/1361462/T1/T1.nii.gz')

        self.image_filenames.remove('/home2/migrate_handan/Biobank_images/fmri_2/2154727/T1/T1.nii.gz')
        self.image_filenames.remove('/home2/migrate_handan/Biobank_images/fmri_2/1356383/T1/T1.nii.gz')

        self.input_transform = input_transform
        self.target_transform = target_transform

    def get_all_file(self):
        filepathlist = []
        for extension in self.endlist:
            filepathlist = filepathlist + glob.glob(self.image_dir + '**/' + extension, recursive=True)
        return filepathlist

    def is_mri_file(self, filename):
        return any(filename.endswith(extension) for extension in self.endlist)


    def __getitem__(self, index):
        mripath = self.image_filenames[index]

        input = loadmri(mripath)

        userid = mripath.split('/')[-3]

        target_age = np.array([self.target_dict[userid]])

        target, bc = dpu.num2vect(target_age, self.bin_range, self.bin_step, self.sigma)
        #target = torch.tensor(target, dtype=torch.float32)
        #print(f'Label shape: {y.shape}')

        # Preprocessing
        input = input / input.mean()
        #print(input.shape)
        input = dpu.crop_center(input, self.input_size)
        #print(input.shape)

        sp_input = (1,) + input.shape
        input = input.reshape(sp_input)

        #sp_target = (1,) + target.shape
        #target = target.reshape(sp_target)

        input = torch.tensor(input, dtype=torch.float32)
        #print(mripath)
        target = torch.tensor(target, dtype=torch.float32)
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target, userid, target_age

    def __len__(self):
        return len(self.image_filenames)


class PCADataProcessor:
    def __init__(self, cfg):
        super(PCADataProcessor, self).__init__()
        self.cfg = cfg
        self.target_dict_path = self.cfg.PCA.TARGET_DICT
        self.train_filelist = list(np.load(cfg.PCA.TRAIN_IMAGELIST_PATH, allow_pickle=True))
        self.test_filelist = list(np.load(cfg.PCA.TEST_IMAGELIST_PATH, allow_pickle=True))

        self.target_dict = np.load(self.target_dict_path, allow_pickle=True).item()
        self.input_size = self.cfg.PCA.CROP_SIZE
        self.all_train_image_filenames = self.train_filelist
        self.test_image_filenames = self.test_filelist
        self.model_save_path = cfg.PCA.MODEL_SAVE_PATH
        self.data_save_path = cfg.PCA.DATA_SAVE_PATH
        self.filter_size = cfg.PCA.FILTERED_SIZE
        self.sparse_mat_path = cfg.PCA.SPARSE_SAVE_PATH

        random.shuffle(self.all_train_image_filenames)

        num_all_train_files = len(self.all_train_image_filenames)

        self.train_image_filenames = self.all_train_image_filenames[0:int(num_all_train_files*(1-self.cfg.PCA.TEST_PERCENT))]
        self.validation_image_filenames = self.all_train_image_filenames[int(num_all_train_files*(1-self.cfg.PCA.TEST_PERCENT)):]

        self.num_train_files = len(self.train_image_filenames)
        self.num_validation_files = len(self.validation_image_filenames)
        self.num_test_files = len(self.test_image_filenames)

    def mri2nparray(self, mripath):
        input = loadmri(mripath)
        userid = mripath.split('/')[-3]
        target_age = float(self.target_dict[userid])

        # Preprocessing
        input = input / input.mean()
        # print(input.shape)
        input = dpu.crop_center(input, self.input_size)

        input = skimage.measure.block_reduce(input, self.filter_size, np.max)

        input = input.flatten()

        return input, target_age


    def get_PCA_data(self):
        num_samples = self.num_train_files + self.num_validation_files + self.num_test_files
        pca_dim = self.cfg.PCA.DIMENSION
        len_sample = int(self.input_size[0] * self.input_size[1] * self.input_size[2] / (self.filter_size[0] * self.filter_size[1] * self.filter_size[2]))


        age_array =  np.zeros((num_samples,))

        pca_output_data = np.zeros((num_samples, pca_dim), dtype='float32')

        #pca = PCA(n_components=pca_dim, svd_solver = 'arpack')
        #pca = decomposition.TruncatedSVD(n_components=pca_dim, algorithm='arpack') # svd
        pca = IncrementalPCA(n_components=pca_dim)

        batch_size = 1235#int(pca_dim)#num_samples // 100
        all_image_filenames = self.train_image_filenames + self.validation_image_filenames + self.test_image_filenames
        pca_input_data = np.zeros((batch_size, len_sample), dtype='float32')
        for i in range(num_samples):
            if i % 100 == 0:
                print(f'Training The {i} th MRI...')
            mripath = all_image_filenames[i]
            input_sample, target_age = self.mri2nparray(mripath)
            age_array[i] = target_age
            if i % batch_size == 0 and i >= batch_size and i < num_samples-batch_size:
                print(np.shape(pca_input_data))
                pca.partial_fit(pca_input_data)
                pca_input_data = np.zeros((batch_size, len_sample), dtype='float32')
                pca_input_data[(i % batch_size), :] = input_sample

            elif i % batch_size == 0 and i >= batch_size and i >= num_samples-batch_size:
                pca.partial_fit(pca_input_data)
                if num_samples % batch_size == 0:
                    pca_input_data = np.zeros(((batch_size), len_sample), dtype='float32')
                else:
                    pca_input_data = np.zeros(((num_samples % batch_size), len_sample), dtype='float32')
                pca_input_data[(i % batch_size), :] = input_sample
            elif i == num_samples - 1:
                pca_input_data[(i % batch_size), :] = input_sample
                pca.partial_fit(pca_input_data)
            else:
                pca_input_data[(i % batch_size), :] = input_sample
        # transform
        pca_input_data = np.zeros((batch_size, len_sample), dtype='float32')
        for i in range(num_samples):
            if i % 100 == 0:
                print(f'Transforming the {i} th MRI...')
            mripath = all_image_filenames[i]
            input_sample, target_age = self.mri2nparray(mripath)
            if i % batch_size == 0 and i >= batch_size and i < num_samples-batch_size:
                pca_output_data[(i-batch_size):i, :] = pca.transform(pca_input_data)
                pca_input_data = np.zeros((batch_size, len_sample), dtype='float32')
                pca_input_data[(i % batch_size), :] = input_sample
            elif i % batch_size == 0 and i >= batch_size and i >= num_samples-batch_size:
                pca_output_data[(i-batch_size):i, :] = pca.transform(pca_input_data)
                if num_samples % batch_size == 0:
                    pca_input_data = np.zeros(((batch_size), len_sample), dtype='float32')
                else:
                    pca_input_data = np.zeros(((num_samples % batch_size), len_sample), dtype='float32')
                pca_input_data[(i % batch_size), :] = input_sample
            elif i == num_samples - 1:
                pca_input_data[(i % batch_size), :] = input_sample
                if num_samples % batch_size == 0:
                    pca_output_data[(i - batch_size + 1):, :] = pca.transform(pca_input_data)
                else:
                    pca_output_data[(i-(num_samples % batch_size)+1):, :] = pca.transform(pca_input_data)
            else:
                pca_input_data[(i % batch_size), :] = input_sample


        #pca_output_data = pca.fit_transform(pca_input_data)

        #pickle.dump(pca, open(self.model_save_path, 'wb'))

        train_data = pca_output_data[0:self.num_train_files, :]
        train_age_array = age_array[0:self.num_train_files]

        validation_data = pca_output_data[self.num_train_files: (self.num_train_files+self.num_validation_files),:]
        validation_age_array = age_array[self.num_train_files: (self.num_train_files+self.num_validation_files)]

        test_data = pca_output_data[(self.num_train_files+self.num_validation_files):, :]
        test_age_array = age_array[(self.num_train_files+self.num_validation_files):]

        np.savez(self.data_save_path, train_data, train_age_array, validation_data, validation_age_array, test_data, test_age_array)

        return train_data, train_age_array, validation_data, validation_age_array, test_data, test_age_array

    def get_PCA_data_sparse(self):
        num_samples = self.num_train_files + self.num_validation_files + self.num_test_files
        pca_dim = self.cfg.PCA.DIMENSION
        len_sample = self.input_size[0] * self.input_size[1] * self.input_size[2] / (self.filter_size[0] * self.filter_size[1] * self.filter_size[2])

        age_array =  np.zeros((num_samples,))

        #pca_output_data = np.zeros((num_samples, pca_dim), dtype='float32')

        #pca = PCA(n_components=pca_dim, svd_solver = 'arpack')
        pca = decomposition.TruncatedSVD(n_components=pca_dim, algorithm='arpack') # svd
        #pca = IncrementalPCA(n_components=pca_dim)

        #batch_size = pca_dim#num_samples // 100
        all_image_filenames = self.train_image_filenames + self.validation_image_filenames + self.test_image_filenames
        #pca_input_data = np.zeros((batch_size, len_sample), dtype='float32')
        print('Start transforming MRI data to sparse matrix data...')
        for i in range(num_samples):
            mripath = all_image_filenames[i]
            input_sample, target_age = self.mri2nparray(mripath)
            age_array[i] = target_age
            if i % 100 == 0:
                print(f'{i} samples processed..')
            if i == 0:
                pca_input_data = sp.csr_matrix(input_sample)
            else:
                sparse_input_sample = sp.csr_matrix(input_sample)
                pca_input_data = sp.vstack((pca_input_data, sparse_input_sample))



        print('Start training truncatedSVD...')
        pca_output_data = pca.fit_transform(pca_input_data)
        print('Finished training truncatedSVD!')

        sp.save_npz(self.sparse_mat_path, pca_input_data)

        #pickle.dump(pca, open(self.model_save_path, 'wb'))

        train_data = pca_output_data[0:self.num_train_files, :]
        train_age_array = age_array[0:self.num_train_files]

        validation_data = pca_output_data[self.num_train_files: (self.num_train_files+self.num_validation_files),:]
        validation_age_array = age_array[self.num_train_files: (self.num_train_files+self.num_validation_files)]

        test_data = pca_output_data[(self.num_train_files+self.num_validation_files):, :]
        test_age_array = age_array[(self.num_train_files+self.num_validation_files):]

        np.savez(self.data_save_path, train_data, train_age_array, validation_data, validation_age_array, test_data, test_age_array)

        return train_data, train_age_array, validation_data, validation_age_array, test_data, test_age_array







class DatasetFromFolder_mri_general(data.Dataset):
    def __init__(self, cfg,  image_dir = None, target_dict_path = None, filelist = None, input_transform=None, target_transform=None, direction = 'left_right', mode = 'train_notsave'):
        super(DatasetFromFolder_mri_general, self).__init__()
        self.cfg = cfg
        self.target_dict = np.load(target_dict_path, allow_pickle=True).item()
        self.bin_range = self.cfg.NET_DATALOADER.BIN_RANGE
        self.input_size = self.cfg.NET_INPUT.CROP_SIZE
        self.bin_step = self.cfg.NET_DATALOADER.BIN_STEP
        self.sigma = self.cfg.NET_DATALOADER.SIGMA
        self.direction = direction
        self.dataset = self.cfg.NET_DATALOADER.DATASET
        self.mode = mode

        if filelist is not None:
            self.image_filenames = filelist
        else:
            self.endlist = self.cfg.NET_DATALOADER.ENDLIST
            self.image_dir = image_dir

            self.image_filenames = self.get_all_file()


        if self.dataset == 'ukbiobank':
            removefile_list = ['/home2/migrate_handan/Biobank_images/fmri_2/1361462/T1/T1.nii.gz',
                               '/home2/migrate_handan/Biobank_images/fmri_2/2154727/T1/T1.nii.gz',
                               '/home2/migrate_handan/Biobank_images/fmri_2/1356383/T1/T1.nii.gz']

            for file in removefile_list:
                if file in self.image_filenames:
                    self.image_filenames.remove(file)

        if self.mode == 'train_save':
            random.shuffle(self.image_filenames)

            num_files = len(self.image_filenames)

            self.train_image_filenames = self.image_filenames[0:int(num_files*(1-self.cfg.NET_DATALOADER.TEST_PERCENT))]
            self.test_image_filenames = self.image_filenames[int(num_files*(1-self.cfg.NET_DATALOADER.TEST_PERCENT)):]

            np.save(self.cfg.NET_DATALOADER.TRAIN_IMAGELIST_PATH, self.train_image_filenames)
            np.save(self.cfg.NET_DATALOADER.TEST_IMAGELIST_PATH, self.test_image_filenames)

        elif self.mode == 'train_notsave':
            random.shuffle(self.image_filenames)

            num_files = len(self.image_filenames)

            self.train_image_filenames = self.image_filenames[
                                         0:int(num_files * (1 - self.cfg.NET_DATALOADER.TEST_PERCENT))]
            self.test_image_filenames = self.image_filenames[
                                        int(num_files * (1 - self.cfg.NET_DATALOADER.TEST_PERCENT)):]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def get_all_file(self):
        filepathlist = []
        for extension in self.endlist:
            filepathlist = filepathlist + glob.glob(self.image_dir + '**/' + extension, recursive=True)
        return filepathlist

    def is_mri_file(self, filename):
        return any(filename.endswith(extension) for extension in self.endlist)


    def __getitem__(self, index):
        mripath = self.image_filenames[index]

        #print(mripath)
        input = loadmri(mripath)

        if self.dataset == 'ukbiobank':
            userid = mripath.split('/')[-3]
        elif self.dataset == 'abide':
            userid = mripath.split('/')[-4]
        elif self.dataset == 'oasis':
            userid = mripath.split('/')[-1]

        target_age = np.array([self.target_dict[userid]]).astype(np.float)

        target, bc = dpu.num2vect(target_age, self.bin_range, self.bin_step, self.sigma)
        #target = torch.tensor(target, dtype=torch.float32)
        #print(f'Label shape: {y.shape}')

        # Preprocessing
        input = input / input.mean()
        #print(input.shape)
        input = dpu.crop_center(input, self.input_size)
        #print(input.shape)

        if self.direction == 'left_right':
            input_left = input[0:int(self.input_size[0]/2), :, :]
            input_right = input[int(self.input_size[0]/2):, :, :]


        sp_input_left = (1,) + input_left.shape
        input_left = input_left.reshape(sp_input_left)
        input_left = torch.tensor(input_left, dtype=torch.float32)

        sp_input_right = (1,) + input_right.shape
        input_right = input_right.reshape(sp_input_right)
        input_right = torch.tensor(input_right, dtype=torch.float32)
        #print(mripath)
        sp_input = (1,) + input.shape
        input = input.reshape(sp_input)
        input= torch.tensor(input, dtype=torch.float32)

        target = torch.tensor(target, dtype=torch.float32)
        if self.input_transform:
            input_left = self.input_transform(input_left)
            input_right = self.input_transform(input_right)
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        target_age = torch.Tensor(target_age)

        return input_left, input_right, target, userid, target_age, input

    def __len__(self):
        return len(self.image_filenames)






class DatasetFromFolder_mri_leftright(data.Dataset):
    def __init__(self, cfg,  image_dir = None, target_dict_path = None, filelist = None, input_transform=None, target_transform=None, direction = 'left_right', mode = 'train'):
        super(DatasetFromFolder_mri_leftright, self).__init__()
        self.cfg = cfg
        self.target_dict = np.load(target_dict_path, allow_pickle=True).item()
        self.bin_range = self.cfg.NET_DATALOADER.BIN_RANGE
        self.input_size = self.cfg.NET_INPUT.CROP_SIZE
        self.bin_step = self.cfg.NET_DATALOADER.BIN_STEP
        self.sigma = self.cfg.NET_DATALOADER.SIGMA
        self.direction = direction

        if filelist is not None:
            self.image_filenames = filelist
        else:
            self.endlist = self.cfg.NET_DATALOADER.ENDLIST
            self.image_dir = image_dir

            self.image_filenames = self.get_all_file()


        removefile_list = ['/home2/migrate_handan/Biobank_images/fmri_2/1361462/T1/T1.nii.gz',
                           '/home2/migrate_handan/Biobank_images/fmri_2/2154727/T1/T1.nii.gz',
                           '/home2/migrate_handan/Biobank_images/fmri_2/1356383/T1/T1.nii.gz']

        for file in removefile_list:
            if file in self.image_filenames:
                self.image_filenames.remove(file)

        if mode == 'train_save':
            random.shuffle(self.image_filenames)

            num_files = len(self.image_filenames)

            self.train_image_filenames = self.image_filenames[0:int(num_files*(1-self.cfg.NET_DATALOADER.TEST_PERCENT))]
            self.test_image_filenames = self.image_filenames[int(num_files*(1-self.cfg.NET_DATALOADER.TEST_PERCENT)):]

            np.save(self.cfg.NET_DATALOADER.TRAIN_IMAGELIST_PATH, self.train_image_filenames)
            np.save(self.cfg.NET_DATALOADER.TEST_IMAGELIST_PATH, self.test_image_filenames)

        elif mode == 'train_notsave':
            random.shuffle(self.image_filenames)

            num_files = len(self.image_filenames)

            self.train_image_filenames = self.image_filenames[
                                         0:int(num_files * (1 - self.cfg.NET_DATALOADER.TEST_PERCENT))]
            self.test_image_filenames = self.image_filenames[
                                        int(num_files * (1 - self.cfg.NET_DATALOADER.TEST_PERCENT)):]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def get_all_file(self):
        filepathlist = []
        for extension in self.endlist:
            filepathlist = filepathlist + glob.glob(self.image_dir + '**/' + extension, recursive=True)
        return filepathlist

    def is_mri_file(self, filename):
        return any(filename.endswith(extension) for extension in self.endlist)


    def __getitem__(self, index):
        mripath = self.image_filenames[index]

        #print(mripath)
        input = loadmri(mripath)

        userid = mripath.split('/')[-3]

        target_age = np.array([self.target_dict[userid]]).astype(np.float)

        target, bc = dpu.num2vect(target_age, self.bin_range, self.bin_step, self.sigma)
        #target = torch.tensor(target, dtype=torch.float32)
        #print(f'Label shape: {y.shape}')

        # Preprocessing
        input = input / input.mean()
        #print(input.shape)
        input = dpu.crop_center(input, self.input_size)
        #print(input.shape)

        if self.direction == 'left_right':
            input_left = input[0:int(self.input_size[0]/2), :, :]
            input_right = input[int(self.input_size[0]/2):, :, :]


        sp_input_left = (1,) + input_left.shape
        input_left = input_left.reshape(sp_input_left)
        input_left = torch.tensor(input_left, dtype=torch.float32)

        sp_input_right = (1,) + input_right.shape
        input_right = input_right.reshape(sp_input_right)
        input_right = torch.tensor(input_right, dtype=torch.float32)
        #print(mripath)
        sp_input = (1,) + input.shape
        input = input.reshape(sp_input)
        input= torch.tensor(input, dtype=torch.float32)

        target = torch.tensor(target, dtype=torch.float32)
        if self.input_transform:
            input_left = self.input_transform(input_left)
            input_right = self.input_transform(input_right)
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        target_age = torch.Tensor(target_age)

        return input_left, input_right, target, userid, target_age, input

    def __len__(self):
        return len(self.image_filenames)





class equalNumSamplePicker:
    def __init__(self, cfg):
        super(equalNumSamplePicker, self).__init__()
        self.cfg = cfg
        self.target_dict_path = self.cfg.PCA.TARGET_DICT
        self.train_filelist = list(np.load(cfg.PCA.TRAIN_IMAGELIST_PATH, allow_pickle=True))
        self.test_filelist = list(np.load(cfg.PCA.TEST_IMAGELIST_PATH, allow_pickle=True))

        self.target_dict = np.load(self.target_dict_path, allow_pickle=True).item()
        self.input_size = self.cfg.PCA.CROP_SIZE
        self.all_train_image_filenames = self.train_filelist
        self.test_image_filenames = self.test_filelist
        self.model_save_path = cfg.PCA.MODEL_SAVE_PATH
        self.data_save_path = cfg.PCA.DATA_SAVE_PATH

        random.shuffle(self.all_train_image_filenames)

        num_all_train_files = len(self.all_train_image_filenames)

        self.train_image_filenames = self.all_train_image_filenames[
                                     0:int(num_all_train_files * (1 - self.cfg.PCA.TEST_PERCENT))]
        self.validation_image_filenames = self.all_train_image_filenames[
                                          int(num_all_train_files * (1 - self.cfg.PCA.TEST_PERCENT)):]

        self.num_train_files = len(self.train_image_filenames)
        self.num_validation_files = len(self.validation_image_filenames)
        self.num_test_files = len(self.test_image_filenames)






from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision




class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        print('----------------')
        #df["label"] = self._get_labels(dataset)
        df["label"] = [x[4][0] for x in dataset]

        print('label length:', len(df["label"]))

        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / (label_to_count[df["label"]]) + 1#1/2 # how about add a number
        #for i, wi in weights:


        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[4][0] for x in dataset]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset[:][4][0]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset[:][4][0]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples



