import numpy as np
import sys
sys.path.append('../')
from statistic_config import pca_config
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
pca_cfg = pca_config._C
import csv
import time

predict_save_path = '../../statistic_methods_predict/logistic/predict_data.npz'
csv_save_path = '../../statistic_methods_predict/logistic/ukbiobank'


pca_data_path = pca_cfg.PCA.DATA_SAVE_PATH

whole_data = np.load(pca_data_path+'.npz', allow_pickle=True)

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

clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, max_iter=1000))
clf.fit(train_data, train_age_array)

val_score = clf.score(validation_data, validation_age_array)
print(f'Validation accuracy score = {val_score}.')
predict_validate = clf.predict(validation_data)

test_score = clf.score(test_data, test_age_array)
print(f'test accuracy score = {test_score}.')
predict_test = clf.predict(test_data)


train_score = clf.score(train_data, train_age_array)
print(f'train accuracy score = {train_score}.')
predict_train = clf.predict(train_data)

predict_file = csv_save_path + time.strftime("%Y-%m-%d") + '_age_predict.csv'
predict_file = open(predict_file, "w")  # 创建csv文件
writer = csv.writer(predict_file)
writer.writerow(['user id', 'predict age', 'real age', 'age difference'])
for i in range(len(predict_test)):
    userid = i
    age_x = predict_test[i]
    target_age = test_age_array[i]
    age_difference = age_x - target_age
    writer.writerow([userid, age_x, target_age, age_difference])

predict_file.close()

np.savez(predict_save_path,  predict_train, train_age_array, predict_test, test_age_array, predict_validate, validation_age_array)






