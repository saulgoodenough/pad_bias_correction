import numpy as np
import sys
sys.path.append('../')
from statistic_config import pca_config, pca_sparse_config
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import sparse as sp

from sklearn.svm import SVR
import xgboost as xgb

from utils.logger import setup_logger

from sklearn import metrics
from sklearn.metrics import mean_absolute_error


import csv
import time



pca_cfg = pca_config._C

pca_sparse_config = pca_sparse_config._C

predict_save_path = '../../statistic_methods_predict/xgboost/predict_data.npz'
csv_save_path = '../../statistic_methods_predict/xgboost/ukbiobank'
model_save_dir = '../../statistic_methods_predict/xgboost/'

logger = setup_logger("xgboost sparse", model_save_dir, if_train=True)

pca_data_path = pca_cfg.PCA.DATA_SAVE_PATH

#sparse_matrix_data = sp.load_npz(pca_sparse_config.PCA.SPARSE_SAVE_PATH)

train_num = 7979
validate_num = 419
test_num = 1482

whole_data = np.load(pca_data_path, allow_pickle=True)
#whole_data = np.load(pca_sparse_config.PCA.DATA_SAVE_PATH, allow_pickle=True)
#whole_data = np.load(pca_sparse_config.PCA.COMBINE_DATA_SAVE_PATH, allow_pickle=True)

train_data = whole_data["arr_0"]
train_age_array = whole_data["arr_1"]
logger.info(f'Training sample amount = {len(train_age_array)}')

validation_data = whole_data["arr_2"]
validation_age_array = whole_data["arr_3"]
logger.info(f'Validation sample amount = {len(validation_age_array)}')

test_data = whole_data["arr_4"]
test_age_array = whole_data["arr_5"]
logger.info(f'Test sample amount = {len(test_age_array)}')


#train_data = sparse_matrix_data[0:train_num, :]
#validation_data = sparse_matrix_data[train_num:(train_num+validate_num), :]
#test_data = sparse_matrix_data[(train_num+validate_num):, :]


clf= xgb.XGBRegressor(n_estimators=5000,
                      max_depth=7,
                      booster = 'gblinear',
                      eta=0.1,
                      subsample=0.2,
                      gamma = 0.01,
                      colsample_bytree=0.8,
                      min_child_weight=5,
                      verbosity=2,
                      reg_lambda= 10,
                      reg_alpha = 0,
                      silent=0,
                      learning_rate = 0.1,
                      seed=1000,
                      nthread=8)
#train_data = xgb.DMatrix(train_data, label=train_age_array, missing=np.NaN)
#test_data = xgb.DMatrix(test_data,  missing=np.NaN)


clf.fit(train_data, train_age_array, eval_metric = 'mae')


predict_validate = clf.predict(validation_data)
print(f'Validate MAE = {mean_absolute_error(predict_validate, validation_age_array)}')

predict_test = clf.predict(test_data)
print(f'Test MAE = {mean_absolute_error(predict_test, test_age_array)}')


predict_train = clf.predict(train_data)
print(f'Train MAE = {mean_absolute_error(predict_train, train_age_array)}')


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





