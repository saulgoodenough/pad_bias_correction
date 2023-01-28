from scipy import sparse as sp

import numpy as np

import sys
sys.path.append('../')
from statistic_config import pca_config, pca_sparse_config


import csv
import time


pca_cfg = pca_config._C
pca_sparse_config = pca_sparse_config._C

pca_data_path = pca_cfg.PCA.DATA_SAVE_PATH

#sparse_matrix_data = sp.load_npz(pca_sparse_config.PCA.SPARSE_SAVE_PATH)

train_num = 7979
validate_num = 419
test_num = 1482


whole_data = np.load(pca_sparse_config.PCA.DATA_SAVE_PATH, allow_pickle=True)


train_data = whole_data["arr_0"]
print(np.shape(train_data))
train_age_array = whole_data["arr_1"]

validation_data = whole_data["arr_2"]
print(np.shape(validation_data))

val_len = np.shape(validation_data)[0]

validation_age_array = whole_data["arr_3"]
test_data = whole_data["arr_4"]
print(np.shape(test_data))
test_age_array = whole_data["arr_5"]

train_data_add = validation_data[0:(val_len - validate_num), :]
train_age_array_add = validation_age_array[0:(val_len - validate_num)]
train_data = np.concatenate((train_data, train_data_add), axis=0)
train_age_array = np.concatenate((train_age_array, train_age_array_add), axis=0)

validate_data_new = validation_data[(val_len - validate_num):, :]
validate_age_array_new = validation_age_array[(val_len - validate_num):]

np.savez(pca_sparse_config.PCA.DATA_SAVE_PATH, train_data, train_age_array, validate_data_new, validate_age_array_new, test_data, test_age_array)

