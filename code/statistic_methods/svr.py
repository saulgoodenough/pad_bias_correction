import numpy as np
import sys
sys.path.append('../')
from statistic_config import pca_config, pca_sparse_config
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR

from utils.logger import setup_logger


import csv
import time


pca_cfg = pca_config._C

pca_sparse_config = pca_sparse_config._C

predict_save_path = '../../statistic_methods_predict/svr/predict_data.npz'
csv_save_path = '../../statistic_methods_predict/svr/ukbiobank'
model_save_dir = '../../statistic_methods_predict/svr/'

logger = setup_logger("3d net model", model_save_dir, if_train=True)

pca_data_path = pca_cfg.PCA.DATA_SAVE_PATH

whole_data = np.load(pca_data_path, allow_pickle=True)
#whole_data = np.load(pca_sparse_config.PCA.DATA_SAVE_PATH, allow_pickle=True)
#whole_data = np.load(pca_sparse_config.PCA.COMBINE_DATA_SAVE_PATH, allow_pickle=True)


print(len(whole_data))
print(whole_data)
for i in whole_data:
    print(i)

train_data = whole_data["arr_0"]
print(np.shape(train_data))
train_age_array = whole_data["arr_1"]
validation_data = whole_data["arr_2"]
print(np.shape(validation_data))
validation_age_array = whole_data["arr_3"]
test_data = whole_data["arr_4"]
print(np.shape(test_data))
test_age_array = whole_data["arr_5"]


clf = make_pipeline(StandardScaler(), SVR(degree=6, C=1e2, epsilon = 0.01, tol = 1e-6, gamma='auto', cache_size=1000, kernel='linear'))
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






