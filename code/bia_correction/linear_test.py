import numpy as np

from code.utils.bias_correction import split_validate_train as split_validate_train
from code.utils.bias_correction import linear_correction as linear_correction


# resnet range
whole_path = '../../resnet3d_predict/range/whole/2021-08-04_age_predict.csv'
age_true_validate, age_predict_validate, age_true_test, age_predict_test = split_validate_train(whole_path)

a,b,corrected_predict_age = linear_correction(age_true_validate, age_predict_validate, age_true_test, age_predict_test)


print(a,b,corrected_predict_age)

# resnet range sampler

