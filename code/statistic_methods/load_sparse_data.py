from scipy import sparse as sp

import numpy as np

import sys
sys.path.append('../')
from statistic_config import pca_config, pca_sparse_config


import csv
import time


pca_cfg = pca_config._C

pca_data_path = pca_cfg.PCA.DATA_SAVE_PATH

sparse_matrix_data = sp.load_npz(pca_sparse_config.PCA.SPARSE_SAVE_PATH)

train_num = 7978
validate_num = 419
test_num = 1482

whole_data = np.load(pca_data_path+'.npz', allow_pickle=True)

train_age_array = whole_data["arr_1"]
print(f'Training sample amount = {len(train_age_array)}')

validation_age_array = whole_data["arr_3"]
print(f'Validation sample amount = {len(validation_age_array)}')

test_age_array = whole_data["arr_5"]
print(f'Test sample amount = {len(test_age_array)}')

train_data = sparse_matrix_data[0:train_num, :]
validation_data = sparse_matrix_data[train_num:(train_num+validate_num), :]
test_data = sparse_matrix_data[(train_num+validate_num):, :]









