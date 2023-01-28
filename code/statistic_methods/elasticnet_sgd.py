import numpy as np
import sys
sys.path.append('../')
from statistic_config import pca_config, pca_sparse_config
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import sparse as sp

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVR

from utils.logger import setup_logger


import csv
import time


pca_cfg = pca_config._C

pca_sparse_config = pca_sparse_config._C

predict_save_path = '../../statistic_methods_predict/elasticnet_sgd/predict_data.npz'
csv_save_path = '../../statistic_methods_predict/elasticnet_sgd/ukbiobank'
model_save_dir = '../../statistic_methods_predict/elasticnet_sgd/'

logger = setup_logger("elasticnet sgd sparse", model_save_dir, if_train=True)

pca_data_path = pca_cfg.PCA.DATA_SAVE_PATH

sparse_matrix_data = sp.load_npz(pca_sparse_config.PCA.SPARSE_SAVE_PATH)

train_num = 7979
validate_num = 419
test_num = 1482

whole_data = np.load(pca_data_path+'.npz', allow_pickle=True)

train_age_array = whole_data["arr_1"]
logger.info(f'Training sample amount = {len(train_age_array)}')

validation_age_array = whole_data["arr_3"]
logger.info(f'Validation sample amount = {len(validation_age_array)}')

test_age_array = whole_data["arr_5"]
logger.info(f'Test sample amount = {len(test_age_array)}')

train_data = sparse_matrix_data[0:train_num, :]
validation_data = sparse_matrix_data[train_num:(train_num+validate_num), :]
test_data = sparse_matrix_data[(train_num+validate_num):, :]

clf = make_pipeline(StandardScaler(with_mean=False), SGDRegressor(random_state=0, penalty='elasticnet', max_iter=2000))
clf.fit(train_data, train_age_array)
val_score = clf.score(validation_data, validation_age_array)
logger.info(f'Validation score = {val_score}.')
predict_validate = clf.predict(validation_data)

test_score = clf.score(test_data, test_age_array)
logger.info(f'test score = {test_score}.')
predict_test = clf.predict(test_data)


train_score = clf.score(train_data, train_age_array)
logger.info(f'train score = {train_score}.')
predict_train = clf.predict(train_data)

predict_file = csv_save_path + time.strftime("%Y-%m-%d") + '_age_predict.csv'
predict_file = open(predict_file, "w")  # 创建csv文件
writer = csv.writer(predict_file)
writer.writerow(['user id', 'predict age', 'real age', 'age difference'])
mean_diff = 0
for i in range(len(predict_test)):
    userid = i
    age_x = predict_test[i]
    target_age = test_age_array[i]
    age_difference = age_x - target_age
    mean_diff += abs(age_difference)
    writer.writerow([userid, age_x, target_age, age_difference])

predict_file.close()
logger.info(f'Mean age difference = { mean_diff/len(predict_test)}')

np.savez(predict_save_path,  predict_train, train_age_array, predict_test, test_age_array, predict_validate, validation_age_array)






