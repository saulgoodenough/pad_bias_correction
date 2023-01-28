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

whole_data = np.load(pca_sparse_config.PCA.DATA_SAVE_PATH, allow_pickle=True)

train_num = 7979
validate_num = 419
test_num = 1482

#whole_data = np.load(pca_data_path+'.npz', allow_pickle=True)

train_data = whole_data["arr_0"]
train_age_array = whole_data["arr_1"]
print(f'Training sample amount = {len(train_age_array)}')

validation_data = whole_data["arr_2"]
validation_age_array = whole_data["arr_3"]
print(f'Validation sample amount = {len(validation_age_array)}')

test_data = whole_data["arr_4"]
test_age_array = whole_data["arr_5"]
print(f'Test sample amount = {len(test_age_array)}')

#train_data = whole_data[0:train_num, :]
#validation_data = whole_data[train_num:(train_num+validate_num), :]
#test_data = whole_data[(train_num+validate_num):, :]









