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


whole_data_1 = np.load(pca_cfg.PCA.DATA_SAVE_PATH, allow_pickle=True)

whole_data_2 = np.load(pca_sparse_config.PCA.DATA_SAVE_PATH, allow_pickle=True)


train_data_1 = whole_data_1["arr_0"]
print(np.shape(train_data_1))
train_age_array = whole_data_1["arr_1"]

validation_data_1 = whole_data_1["arr_2"]
print(np.shape(validation_data_1))

val_len = np.shape(validation_data_1)[0]

validation_age_array = whole_data_1["arr_3"]
test_data_1 = whole_data_1["arr_4"]
print(np.shape(test_data_1))
test_age_array = whole_data_1["arr_5"]


train_data_2 = whole_data_2["arr_0"]
validation_data_2 = whole_data_2["arr_2"]
test_data_2 = whole_data_2["arr_4"]


train_data = np.concatenate((train_data_1, train_data_2), axis=1)
validation_data = np.concatenate((validation_data_1, validation_data_2), axis=1)
test_data = np.concatenate((test_data_1, test_data_2), axis=1)

print(np.shape(train_data), np.shape(validation_data), np.shape(test_data))

np.savez(pca_sparse_config.PCA.COMBINE_DATA_SAVE_PATH, train_data, train_age_array, validation_data, validation_age_array, test_data, test_age_array)
