import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import sys
from sklearn.svm import SVR
import sklearn.pipeline as pl
import sklearn.linear_model as lm
import sklearn.preprocessing as sp
import os

def split_validate_train(whole_path, ifval = False):
    whole_pd = pd.read_csv(whole_path)

    #
    val_ratio = 0.25
    if ifval == True:

        val_len = int(whole_pd.shape[0] * val_ratio)

        validate_pd = whole_pd[0:val_len]
        test_pd = whole_pd[val_len+1:]

        age_true_validate = validate_pd['real age'].values
        age_predict_validate = validate_pd['predict age'].values
        len_validate = len(age_true_validate)
        age_true_validate = np.reshape(age_true_validate, (len_validate,1))
        age_predict_validate = np.reshape(age_predict_validate, (len_validate,1))

        age_true_test = test_pd['real age'].values
        age_predict_test = test_pd['predict age'].values
        len_test = len(age_true_test)
        age_true_test = np.reshape(age_true_test, (len_test,1))
        age_predict_test = np.reshape(age_predict_test, (len_test,1))
    else:
        #val_ratio = 0.25
        #val_len = int(whole_pd.shape[0] * val_ratio)

        test_pd = whole_pd

        age_true_test = test_pd['real age'].values
        age_predict_test = test_pd['predict age'].values
        len_test = len(age_true_test)
        age_true_test = np.reshape(age_true_test, (len_test, 1))
        age_predict_test = np.reshape(age_predict_test, (len_test, 1))


    return age_true_test, age_predict_test, age_true_test, age_predict_test, test_pd


def linear_correction(age_true_validate, age_predict_validate, age_true_test, age_predict_test):
    '''
    :param age_true_validate: chronological age used to get linear regression parameters
    :param age_predict_validate: predict age used to get linear regression parameters
    :param age_true_test: chronological age for test
    :param age_predict_test: predict age for test
    :return: corrected age for test
    age_true_validate = np.array([1,  2, 2,  3])
    age_predict_validate = np.dot(age_predict_validate, np.array([2])) + 3
    '''
    reg = LinearRegression().fit(age_true_validate, age_predict_validate)
    #reg = LinearRegression().fit(age_predict_validate, age_true_validate - age_predict_validate)
    a = reg.coef_
    b = reg.intercept_
    corrected_predict_age = (age_predict_test - b) / a
    #corrected_predict_age = reg.predict(age_predict_test)
    data_len = np.shape(corrected_predict_age)[0]
    corrected_predict_age = np.reshape(corrected_predict_age, (data_len, )) #+np.reshape(age_predict_test, (data_len, ))
    return a,b,corrected_predict_age


def method2_linear_correction(age_true_validate, age_predict_validate, age_true_test, age_predict_test):
    '''
    :param age_true_validate: chronological age used to get linear regression parameters
    :param age_predict_validate: predict age used to get linear regression parameters
    :param age_true_test: chronological age for test
    :param age_predict_test: predict age for test
    :return: corrected age for test
    age_true_validate = np.array([1,  2, 2,  3])
    age_predict_validate = np.dot(age_predict_validate, np.array([2])) + 3
    '''
    reg = LinearRegression().fit(age_true_validate, age_predict_validate - age_true_validate)
    #reg = LinearRegression().fit(age_predict_validate, age_true_validate - age_predict_validate)
    a = reg.coef_
    b = reg.intercept_
    corrected_predict_age = age_predict_test - (a * age_true_test + b)
    #corrected_predict_age = reg.predict(age_predict_test)
    data_len = np.shape(corrected_predict_age)[0]
    corrected_predict_age = np.reshape(corrected_predict_age, (data_len, )) #+np.reshape(age_predict_test, (data_len, ))
    return a,b,corrected_predict_age

def method2_square_correction(age_true_validate, age_predict_validate, age_true_test, age_predict_test):
    '''
    :param age_true_validate: chronological age used to get linear regression parameters
    :param age_predict_validate: predict age used to get linear regression parameters
    :param age_true_test: chronological age for test
    :param age_predict_test: predict age for test
    :return: corrected age for test
    age_true_validate = np.array([1,  2, 2,  3])
    age_predict_validate = np.dot(age_predict_validate, np.array([2])) + 3
    '''
    data_len_validate = np.shape(age_predict_validate)[0]
    age_true_validate = np.reshape(age_true_validate, (data_len_validate,))
    age_predict_validate = np.reshape(age_predict_validate, (data_len_validate,))
    data_len_test = np.shape(age_predict_test)[0]
    age_predict_test = np.reshape(age_predict_test, (data_len_test,))
    # print(np.shape(age_true_validate), np.shape(age_predict_validate))
    coffs = np.polyfit(age_true_validate, age_predict_validate - age_true_validate, 2)
    # print(coffs)
    a = coffs[0]
    # print('a', a)
    # print('age_predict_test', age_predict_test)
    b = coffs[1]
    c = coffs[2]
    #corrected_predict_age = age_predict_test - (-b + (abs(b**2 - 4*a*(c-age_true_test)))**(1/2))/(2*a)
    corrected_predict_age = (-(b+1) + (abs((b+1) ** 2 - 4 * a * (c - age_predict_test))) ** (1 / 2)) / (2 * a)

    # age_predict_test =
    # print()
    return coffs, corrected_predict_age


def square_correction(age_true_validate, age_predict_validate, age_true_test, age_predict_test):
    '''
    :param age_true_validate: chronological age used to get linear regression parameters
    :param age_predict_validate: predict age used to get linear regression parameters
    :param age_true_test: chronological age for test
    :param age_predict_test: predict age for test
    :return: corrected age for test
    age_true_validate = np.array([1,  2, 2,  3])
    age_predict_validate = np.dot(age_predict_validate, np.array([2])) + 3
    '''
    data_len_validate = np.shape(age_predict_validate)[0]
    age_true_validate = np.reshape(age_true_validate, (data_len_validate, ))
    age_predict_validate = np.reshape(age_predict_validate, (data_len_validate, ))
    data_len_test = np.shape(age_predict_test)[0]
    age_predict_test = np.reshape(age_predict_test, (data_len_test, ))
    #print(np.shape(age_true_validate), np.shape(age_predict_validate))
    coffs = np.polyfit(age_true_validate, age_predict_validate, 2)
    #print(coffs)
    a = coffs[0]
    #print('a', a)
    #print('age_predict_test', age_predict_test)
    b = coffs[1]
    c = coffs[2]
    corrected_predict_age = (-b + (abs(b**2 - 4*a*(c-age_predict_test)))**(1/2))/(2*a)
    #age_predict_test =
    #print()
    return coffs, corrected_predict_age


def threeOrder_correction(age_true_validate, age_predict_validate, age_true_test, age_predict_test):
    '''
    :param age_true_validate: chronological age used to get linear regression parameters
    :param age_predict_validate: predict age used to get linear regression parameters
    :param age_true_test: chronological age for test
    :param age_predict_test: predict age for test
    :return: corrected age for test
    age_true_validate = np.array([1,  2, 2,  3])
    age_predict_validate = np.dot(age_predict_validate, np.array([2])) + 3
    '''
    data_len_validate = np.shape(age_predict_validate)[0]
    age_true_validate = np.reshape(age_true_validate, (data_len_validate, ))
    age_predict_validate = np.reshape(age_predict_validate, (data_len_validate, ))
    data_len_test = np.shape(age_predict_test)[0]
    age_predict_test = np.reshape(age_predict_test, (data_len_test, ))
    #print(np.shape(age_true_validate), np.shape(age_predict_validate))
    coffs = np.polyfit(age_true_validate, age_predict_validate, 3)
    #print(coffs)
    age_range_list = [int(x) for x in range(38, 86)]
    corrected_predict_age = np.zeros(np.shape(age_predict_test))
    for k in range(len(age_predict_test)):
        coeff = list(coffs)
        coeff[-1] = coffs[-1] - age_predict_test[k]
        roots = np.roots(tuple(coeff))
        for r in roots:
            if np.iscomplex(r)==False and r <= (max(age_range_list)+2) and r >= (min(age_range_list)-2):
                corrected_predict_age[k] = r
            #age_predict_test =
    #print()
    return coffs, corrected_predict_age


def fourOrder_correction(age_true_validate, age_predict_validate, age_true_test, age_predict_test):
    '''
    :param age_true_validate: chronological age used to get linear regression parameters
    :param age_predict_validate: predict age used to get linear regression parameters
    :param age_true_test: chronological age for test
    :param age_predict_test: predict age for test
    :return: corrected age for test
    age_true_validate = np.array([1,  2, 2,  3])
    age_predict_validate = np.dot(age_predict_validate, np.array([2])) + 3
    '''
    data_len_validate = np.shape(age_predict_validate)[0]
    age_true_validate = np.reshape(age_true_validate, (data_len_validate, ))
    age_predict_validate = np.reshape(age_predict_validate, (data_len_validate, ))
    data_len_test = np.shape(age_predict_test)[0]
    age_predict_test = np.reshape(age_predict_test, (data_len_test, ))
    #print(np.shape(age_true_validate), np.shape(age_predict_validate))
    coffs = np.polyfit(age_true_validate, age_predict_validate, 4)
    #print(coffs)
    age_range_list = [int(x) for x in range(38, 86)]
    corrected_predict_age = np.zeros(np.shape(age_predict_test))
    for k in range(len(age_predict_test)):
        coeff = list(coffs)
        coeff[-1] = coffs[-1] - age_predict_test[k]
        roots = np.roots(tuple(coeff))
        for r in roots:
            if np.iscomplex(r)==False and r <= (max(age_range_list)+2) and r >= (min(age_range_list)-2):
                corrected_predict_age[k] = r
            #age_predict_test =
    #print()
    return coffs, corrected_predict_age


def polynomial_correction(age_true_validate, age_predict_validate, age_true_test, age_predict_test):
    '''
    :param age_true_validate: chronological age used to get linear regression parameters
    :param age_predict_validate: predict age used to get linear regression parameters
    :param age_true_test: chronological age for test
    :param age_predict_test: predict age for test
    :return: corrected age for test
    age_true_validate = np.array([1,  2, 2,  3])
    age_predict_validate = np.dot(age_predict_validate, np.array([2])) + 3
    '''
    poly = sp.PolynomialFeatures(3)
    X = poly.fit_transform(age_predict_validate)
    reg = LinearRegression().fit(X, age_true_validate)
    a = reg.coef_
    b = reg.intercept_
    corrected_predict_age = reg.predict(poly.fit_transform(age_predict_test))
    data_len = np.shape(corrected_predict_age)[0]
    corrected_predict_age = np.reshape(corrected_predict_age, (data_len, ))
    #age_predict_test =
    #print()
    return a,b,corrected_predict_age


def svr_correction(age_true_validate, age_predict_validate, age_true_test, age_predict_test):
    '''
    :param age_true_validate: chronological age used to get linear regression parameters
    :param age_predict_validate: predict age used to get linear regression parameters
    :param age_true_test: chronological age for test
    :param age_predict_test: predict age for test
    :return: corrected age for test
    age_true_validate = np.array([1,  2, 2,  3])
    age_predict_validate = np.dot(age_predict_validate, np.array([2])) + 3
    '''
    reg = SVR(C=1.0, epsilon=0.2)
    reg.fit(age_predict_validate, age_true_validate)
    a = reg.dual_coef_
    #b = reg.intercept_
    corrected_predict_age = reg.predict(age_predict_test)
    data_len = np.shape(corrected_predict_age)[0]
    corrected_predict_age = np.reshape(corrected_predict_age, (data_len, ))
    return a, corrected_predict_age


def correct_age(whole_path, save_path, method = 'linear'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    age_true_validate, age_predict_validate, age_true_test, age_predict_test, test_pd = split_validate_train(whole_path, ifval = False)

    if method == 'linear':
        a, b, corrected_predict_age = linear_correction(age_true_validate, age_predict_validate, age_true_test,
                                                        age_predict_test)
        test_pd['predict age'] = corrected_predict_age
        test_pd['age difference'] = test_pd['real age'] - test_pd['predict age']
        corrected_path = save_path+'linear_corrected.csv'
        test_pd.to_csv(corrected_path, index=False)

        original_stdout = sys.stdout  # Save a reference to the original standard output
        with open(save_path + 'parameters_linear_corrected.txt', 'a+') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print('coefficient:', a)
            print('new bias:',b)
            sys.stdout = original_stdout  # Reset the standard

    elif method == 'method2_linear':
        a, b, corrected_predict_age = method2_linear_correction(age_true_validate, age_predict_validate, age_true_test,
                                                        age_predict_test)
        test_pd['predict age'] = corrected_predict_age
        test_pd['age difference'] = test_pd['real age'] - test_pd['predict age']
        corrected_path = save_path + 'method2_linear_corrected.csv'
        test_pd.to_csv(corrected_path, index=False)

        original_stdout = sys.stdout  # Save a reference to the original standard output
        with open(save_path + 'parameters_method2_linear_corrected.txt', 'a+') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print('coefficient:', a)
            print('new bias:', b)
            sys.stdout = original_stdout  # Reset the standard

    elif method == 'polynomial':
        a, b, corrected_predict_age = polynomial_correction(age_true_validate, age_predict_validate, age_true_test,
                                                        age_predict_test)
        test_pd['predict age'] = corrected_predict_age
        test_pd['age difference'] = test_pd['real age'] - test_pd['predict age']
        corrected_path = save_path + 'polynomial_corrected.csv'
        test_pd.to_csv(corrected_path, index=False)

        original_stdout = sys.stdout  # Save a reference to the original standard output
        with open(save_path + 'parameters_polynomial_corrected.txt', 'a+') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print('coefficient:', a)
            print('new bias:', b)
            sys.stdout = original_stdout  # Reset the standard

    elif method == 'square':
        a, corrected_predict_age = square_correction(age_true_validate, age_predict_validate, age_true_test,
                                                        age_predict_test)
        test_pd['predict age'] = corrected_predict_age
        test_pd['age difference'] = test_pd['real age'] - test_pd['predict age']
        corrected_path = save_path + 'square_corrected.csv'
        test_pd.to_csv(corrected_path, index=False)

        original_stdout = sys.stdout  # Save a reference to the original standard output
        with open(save_path + 'parameters_square_corrected.txt', 'a+') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print('coefficient:', a)
            sys.stdout = original_stdout  # Reset the standard

    elif method == 'method2_square':
        a, corrected_predict_age = method2_square_correction(age_true_validate, age_predict_validate, age_true_test,
                                                        age_predict_test)
        test_pd['predict age'] = corrected_predict_age
        test_pd['age difference'] = test_pd['real age'] - test_pd['predict age']
        corrected_path = save_path + 'method2_square_corrected.csv'
        test_pd.to_csv(corrected_path, index=False)

        original_stdout = sys.stdout  # Save a reference to the original standard output
        with open(save_path + 'parameters_method2_square_corrected.txt', 'a+') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print('coefficient:', a)
            sys.stdout = original_stdout  # Reset the standard


    elif method == 'threeOrder':
        a, corrected_predict_age = threeOrder_correction(age_true_validate, age_predict_validate, age_true_test,
                                                        age_predict_test)
        test_pd['predict age'] = corrected_predict_age
        test_pd['age difference'] = test_pd['real age'] - test_pd['predict age']
        corrected_path = save_path + 'threeOrder_corrected.csv'
        test_pd.to_csv(corrected_path, index=False)

        original_stdout = sys.stdout  # Save a reference to the original standard output
        with open(save_path + 'parameters_threeOrder_corrected.txt', 'a+') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print('coefficient:', a)
            sys.stdout = original_stdout  # Reset the standard

    elif method == 'fourOrder':
        a, corrected_predict_age = fourOrder_correction(age_true_validate, age_predict_validate, age_true_test,
                                                        age_predict_test)
        test_pd['predict age'] = corrected_predict_age
        test_pd['age difference'] = test_pd['real age'] - test_pd['predict age']
        corrected_path = save_path + 'fourOrder_corrected.csv'
        test_pd.to_csv(corrected_path, index=False)

        original_stdout = sys.stdout  # Save a reference to the original standard output
        with open(save_path + 'parameters_fourOrder_corrected.txt', 'a+') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print('coefficient:', a)
            sys.stdout = original_stdout  # Reset the standard


    elif method == 'svr':
        a, corrected_predict_age = svr_correction(age_true_validate, age_predict_validate, age_true_test,
                                                        age_predict_test)
        test_pd['predict age'] = corrected_predict_age
        test_pd['age difference'] = test_pd['real age'] - test_pd['predict age']
        corrected_path = save_path + 'svr_corrected.csv'
        test_pd.to_csv(corrected_path, index=False)

        original_stdout = sys.stdout  # Save a reference to the original standard output
        with open(save_path + 'parameters_svr_corrected.txt', 'a+') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print('coefficient:', a)
            #print('new bias:', b)
            sys.stdout = original_stdout  # Reset the standard
    elif method == None:
        corrected_path = save_path + 'not_corrected.csv'
        test_pd['age difference'] = test_pd['real age'] - test_pd['predict age']
        test_pd.to_csv(corrected_path, index=False)

    return corrected_path