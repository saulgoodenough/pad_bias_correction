import pandas as pd
import scipy.stats

'''
import pylab
import scipy.stats as stats

measurements = np.random.normal(loc = 20, scale = 5, size=100)
print(np.shape(measurements))
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()
'''
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statistics

from matplotlib import rcParams
import matplotlib
matplotlib.use('Agg')

import statsmodels.api as sm

TINY_SIZE = 39
SMALL_SIZE = 42
MEDIUM_SIZE = 46
BIGGER_SIZE = 46

plt.rc('font', size=35)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams["legend.frameon"] = False
#plt.rc('legend',**{'fontsize':16})


rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False

rcParams["axes.linewidth"] = 2

img_width = 10
img_height = 9


def reprocess_before_plot(x_pos, age_diff_avg, labels):
    len_vec = len(x_pos)
    new_x_pos = list(x_pos)
    new_age_diff_avg = list(age_diff_avg)
    new_labels = list(labels)

    while new_age_diff_avg[0] == 0:
        new_x_pos.pop(0)
        new_age_diff_avg.pop(0)
        new_labels.pop(0)
    while new_age_diff_avg[-1] == 0:
        new_x_pos.pop(-1)
        new_age_diff_avg.pop(-1)
        new_labels.pop(-1)

    return new_x_pos, new_age_diff_avg, new_labels

'''
[38, 50], [51, 65], [66, 70], [71, 75], [76, 86]
'''



def zerodivide(array1, array2):
    dividearray = np.zeros(array1.shape)
    len_array = array1.shape[0]
    for i in range(len_array):
        if array1[i] != 0:
            dividearray[i] = array1[i]/array2[i]
    return dividearray


def vis(acc, confusion_mat, labels, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cm = confusion_mat
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % p
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    #fig = plt.figure(figsize=figsize)
    #plt.subplot(1, 2, 1)
    '''
    plt.plot(test_accs, 'g')
    plt.grid(True)
    plt.savefig(whole_path + "accuracy.png")
    plt.show()
    '''
    '''
    x_pos = [x for x in range(len(acc))]
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(x_pos,acc, color='green', align='center', width=0.5)
    plt.xlabel("age")
    plt.ylabel("prediction accuracy")
    #plt.title("predict accuracy of different age")
    plt.xticks(x_pos, labels)
    plt.savefig(save_path + 'predict_accuracy.png')
    #plt.subplot(1, 2, 2)
    fig = plt.figure(figsize=(10,12))
    #sns_plot = sns.heatmap(cm,   cmap="Blues") #annot=annot,fmt='',

    cm_array_df = pd.DataFrame(cm, index=labels, columns=labels)
    sns_plot = sns.heatmap(cm, annot=True, annot_kws={"size": 10}) #
    fig = sns_plot.get_figure()
    fig.savefig(save_path + "confusion_mat.png")
    #plt.show()
    '''


def compute_class_accuracy(whole_path, plot_title, save_path, bin_range = [38, 86]):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    whole_pd = pd.read_csv(whole_path)
    real_age = whole_pd['real age'].values
    whole_predict_age = whole_pd['predict age'].values
    age_diff_vec = whole_pd['age difference'].values

    whole_predict_age = np.round(whole_predict_age)
    real_age = np.round(real_age)

    labels = [str(int(x)) for x in range(bin_range[0], bin_range[1])]
    cm = confusion_matrix(real_age, whole_predict_age) #, labels
    cm_labels = np.unique(np.concatenate((real_age, whole_predict_age))).astype(int)
    cm_array = confusion_matrix(np.round(real_age), np.round(whole_predict_age)) #, labels
    cm_array_df = pd.DataFrame(cm_array, index=cm_labels, columns=cm_labels)

    acc = np.divide(cm_array.diagonal(), np.sum(cm_array, axis=1))

    vis(acc, cm, cm_labels, whole_path)

    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(save_path+'class_report.txt', 'a+') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        #print(classification_report(real_age, whole_predict_age, zero_division=1))  # real_age, whole_predict_age
        print('classification accuracy:', accuracy_score(real_age, whole_predict_age))
        #class_accuracy = []
        sys.stdout = original_stdout  # Reset the standard


def kl_count(age_array, age_range):
    dict_count = {}
    for k in range(len(age_range)):
        dict_count[str(int(age_range[k]))] = 0
    print(age_array)
    for i in age_array:
        dict_count[str(int(i))] += 1
    #print(dict_count)
    #print(dict_count.values())
    proba_array = np.array(list(dict_count.values()))/len(age_array)
    return proba_array

#def filter_by_age_amount(real_age_vec, predict_age_vec, new_age_range_ids):

def num_count(real_age_array, age_range):
    dict_count = {}
    for k in range(len(age_range)):
        dict_count[str(int(age_range[k]))] = 0
    for i in real_age_array:
        dict_count[str(int(i))] += 1
    dict_count_rank = {k: v for k, v in reversed(sorted(dict_count.items(), key=lambda item: item[1]))}
    del_list = []
    for k,v in dict_count_rank.items():
        if v == 0:
            del_list.append(k)
    for k in del_list:
        del dict_count_rank[k]
    new_age_range_ids = list(dict_count_rank.keys())
    return new_age_range_ids, list(dict_count_rank.values())


def kl_count_ids(age_array, age_range_ids):
    dict_count = {}
    for k in age_range_ids:
        dict_count[str(int(k))] = 0
    for i in age_array:
        dict_count[str(int(i))] += 1
    #print(dict_count)
    #print(dict_count.values())
    proba_array = np.array(list(dict_count.values()))/len(age_array)
    return proba_array


def KL_divergence(x,y):
    x_add = 1e-16
    y_add = 1e-16
    kl_value = np.sum(x[i] * np.log2((x[i]+x_add)/(y[i]+y_add)) for i in range(len(x)))
    #print(kl_value)
    return kl_value

def compute_mse(new_real_age_vec, new_predict_age_vec):
    len_vec = len(new_real_age_vec)
    age_diff_vec =  new_predict_age_vec - new_real_age_vec
    #print(age_diff_vec)
    mse_ave_temp = np.matmul(np.matrix(age_diff_vec), np.transpose(np.matrix(age_diff_vec))) / len_vec
    mse_ave = mse_ave_temp.item()
    return mse_ave


def compute_kl_divergence(new_real_age_vec, new_predict_age_vec, age_range):
    KL_prob_real = kl_count(new_real_age_vec, age_range)
    KL_prob_predict = kl_count_ids(age_round_f(new_predict_age_vec, age_range), age_range)
    KL_value = KL_divergence(KL_prob_real, KL_prob_predict)
    return KL_value



def compute_measures_by_rank(real_age_vec, predict_age_vec, age_range):
    new_age_range_ids, age_num_vec = num_count(real_age_vec, age_range)
    print(new_age_range_ids)
    print(age_num_vec)
    age_id_num = len(age_num_vec)
    mse_vec = np.zeros((age_id_num,))
    kl_vec = np.zeros((age_id_num,))
    agediff_vec = np.zeros((age_id_num,))
    agediff_withsign_vec = np.zeros((age_id_num,))
    class_num_vec = np.zeros((age_id_num,))
    sample_num_vec = np.zeros((age_id_num,))
    for i in range(age_id_num):
        age_range_ids = new_age_range_ids[i:]
        #print('------------------------')
        #print(age_range_ids)
        # predict sample amount
        predict_num = sum(age_num_vec[i:])
        sample_num_vec[i] = predict_num
        class_num_vec[i] = age_id_num - i
        new_predict_age_vec = np.array([predict_age_vec[k] for k in range(len(real_age_vec)) if str(int(real_age_vec[k])) in age_range_ids]).astype(np.float)
        new_real_age_vec = np.array([real_age_vec[k] for k in range(len(real_age_vec)) if str(int(real_age_vec[k])) in age_range_ids]).astype(np.float)
        #print(new_real_age_vec)
        #print(new_predict_age_vec)
        agediff_vec[i] = np.mean(np.abs(new_real_age_vec - new_predict_age_vec))
        agediff_withsign_vec[i] = np.mean(new_predict_age_vec - new_real_age_vec)
        mse_vec[i] = compute_mse(new_real_age_vec, new_predict_age_vec)
        kl_vec[i] = compute_kl_divergence(new_real_age_vec, new_predict_age_vec, age_range)

    return class_num_vec, sample_num_vec, agediff_vec, agediff_withsign_vec, mse_vec, kl_vec


def age_round_f(age_array, age_range):
    age_array_rounded = np.round(age_array)
    age_range_list = [int(x) for x in age_range]
    for i in range(len(age_array_rounded)):
        if age_array_rounded[i] < min(age_range_list):
            age_array_rounded[i] = min(age_range_list)
        elif age_array_rounded[i] > max(age_range_list):
            age_array_rounded[i] = max(age_range_list)
    return age_array_rounded


def zscore_PAD(real_age_vec, age_diff_vec):

    unique_real_age_list = list(set(real_age_vec))
    unique_real_age_list.sort(reverse=False)
    mean_dict = {}
    sigma_dict = {}

    temp_list = []
    for real_age in unique_real_age_list:
        for i in range(len(real_age_vec)):
            if real_age_vec[i] == real_age:
                temp_list.append(age_diff_vec[i])
        mean_dict[real_age] = np.mean(temp_list)
        sigma_dict[real_age] = np.std(temp_list)
        temp_list = []


    real_age_vec_zscore = []
    age_diff_vec_zscore = []
    real_age_vec_index = []

    for i in range(len(real_age_vec)):
        real_age = real_age_vec[i]
        age_diff_zscore = (age_diff_vec[i] - mean_dict[real_age])/sigma_dict[real_age]
        if sigma_dict[real_age]!= 0:
            real_age_vec_zscore.append(real_age)
            age_diff_vec_zscore.append(age_diff_zscore)
            real_age_vec_index.append(i)

    return real_age_vec_zscore, age_diff_vec_zscore, real_age_vec_index


def zscore_PAD_endpoint(input_df):
    real_age_vec = input_df['real age']
    age_diff_vec = input_df['age difference']
    predict_age_vec = input_df['predict age']
    unique_real_age_list = list(set(real_age_vec))
    unique_real_age_list.sort(reverse=False)
    mean_dict = {}
    sigma_dict = {}

    temp_list = []
    for real_age in unique_real_age_list:
        for i in range(len(real_age_vec)):
            if real_age_vec[i] == real_age:
                temp_list.append(age_diff_vec[i])
        mean_dict[real_age] = np.mean(temp_list)
        sigma_dict[real_age] = np.std(temp_list)
        temp_list = []


    real_age_vec_zscore = []
    age_diff_vec_zscore = []
    real_age_vec_index = []
    predict_age_zscore = []
    age_diff_vec_new = []

    for i in range(len(real_age_vec)):
        real_age = real_age_vec[i]
        age_diff_zscore = (age_diff_vec[i] - mean_dict[real_age])/sigma_dict[real_age]
        if sigma_dict[real_age]!= 0:
            real_age_vec_zscore.append(real_age)
            age_diff_vec_zscore.append(age_diff_zscore)
            real_age_vec_index.append(input_df['user id'][i])
            predict_age_zscore.append(predict_age_vec[i])
            age_diff_vec_new.append(age_diff_vec[i])

    return real_age_vec_zscore, age_diff_vec_zscore, real_age_vec_index, predict_age_zscore, age_diff_vec_new


def compute_avg(real_age_vec_zscore, age_diff_vec_zscore):
    real_age_zscore_list = list(set(real_age_vec_zscore))
    real_age_zscore_list.sort(reverse=False)
    age_diff_vec_zscore_avg = []

    for real_age in real_age_zscore_list:
        temp_list = []
        for i in range(len(real_age_vec_zscore)):
            if real_age_vec_zscore[i] == real_age:
                temp_list.append(age_diff_vec_zscore[i])
        age_diff_vec_zscore_avg.append(np.mean(temp_list))

    return real_age_zscore_list, age_diff_vec_zscore_avg


def compute_error_all(whole_path, plot_title, save_path, bin_range = [38, 86], if_bar_xlabel = True, if_bar_ylabel = True, if_scatter_xlabel = True, if_scatter_ylabel = True, bar_lim = None):
    #plt.style.use('ggplot')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    whole_pd = pd.read_csv(whole_path)
    real_age_vec = whole_pd['real age'].values
    whole_predict_age = whole_pd['predict age'].values
    #print('whole predict age:', whole_predict_age)

    #age_diff_vec = whole_pd['age difference'].values
    age_diff_vec = whole_predict_age - real_age_vec

    age_range = range(bin_range[0], bin_range[1])
    labels = [str(int(x)) for x in range(bin_range[0], bin_range[1])]

    age_group_dict = {}
    #group_list = [[38, 50], [51, 65], [66, 70], [71, 75], [76, 86]]
    for k in range(bin_range[1]-bin_range[0]):
        age_group_dict[labels[k]] = k

    #print(age_group_dict)
    real_age_num = np.zeros(bin_range[1]-bin_range[0])
    real_age_avg = np.zeros(bin_range[1]-bin_range[0])
    whole_predict_age_avg = np.zeros(bin_range[1]-bin_range[0])
    age_diff_avg = np.zeros(bin_range[1]-bin_range[0])

    age_list_dict = {}
    for k in range(bin_range[1]-bin_range[0]):
        age_list_dict[labels[k]] = []
    agediff_list_dict = {}
    for k in range(bin_range[1]-bin_range[0]):
        agediff_list_dict[labels[k]] = []

    # used to compute whole standard deviation
    agediff_list_stddev_dict = {}
    age_list_stddev_dict = {}
    for k in range(bin_range[1]-bin_range[0]):
        age_list_stddev_dict[labels[k]] = []
        agediff_list_stddev_dict[labels[k]] = []


    pick_sample_num = 20


    for i in range(len(whole_predict_age)):
        user_id, predict_age, real_age, age_diff = whole_pd.iloc[i]
        # print(user_id,predict_age,real_age,age_diff)
        group_id = age_group_dict[str(int(real_age))]

        age_list_stddev_dict[str(int(real_age))].append(predict_age)
        if len(age_list_dict[str(int(real_age))]) <= pick_sample_num:
            age_list_dict[str(int(real_age))].append(predict_age)
        whole_predict_age_avg[group_id] += predict_age

        real_age_avg[group_id] += real_age
        real_age_num[group_id] += 1

        age_diff_avg[group_id] += predict_age - real_age

        agediff_list_stddev_dict[str(int(real_age))].append(predict_age - real_age)

        # age diff list for computing median, maximum and minimum
        if len(agediff_list_dict[str(int(real_age))]) <= pick_sample_num:
            agediff_list_dict[str(int(real_age))].append(predict_age - real_age)
    #print(age_list_dict)

    # compute stardard
    standard_list = []
    variance_list = []
    mean_list = []
    median_list = []
    maximum_list = []
    minimum_list = []
    x_pos_adjust = []
    label_adjust = []
    for k in range(bin_range[1]-bin_range[0]):
        temp_age_list = age_list_dict[labels[k]]
        if len(temp_age_list) >= pick_sample_num:
            standard_list.append(np.std(temp_age_list))
            variance_list.append(np.var(temp_age_list))

        temp_agediff_list = agediff_list_dict[labels[k]]
        if len(temp_agediff_list) >= pick_sample_num:#temp_agediff_list != []:
            x_pos_adjust.append(k)
            label_adjust.append(labels[k])
            mean_list.append(np.mean(temp_agediff_list))
            median_list.append(statistics.median(temp_agediff_list))
            maximum_list.append(max(temp_agediff_list))
            minimum_list.append(min(temp_agediff_list))

    standard_list = []
    std_x_pos_adjust = []
    std_label_adjust = []
    std_pick_sample_num = 5
    for k in range(bin_range[1]-bin_range[0]):
        temp_age_list = age_list_stddev_dict[labels[k]]
        if len(temp_age_list) >= std_pick_sample_num:
            standard_list.append(np.std(temp_age_list))

        temp_agediff_list = agediff_list_stddev_dict[labels[k]]
        if len(temp_agediff_list) >= pick_sample_num:  # temp_agediff_list != []:
            std_x_pos_adjust.append(k)
            std_label_adjust.append(labels[k])


    real_age_avg = zerodivide(real_age_avg,real_age_num)
    whole_predict_age_avg = zerodivide(whole_predict_age_avg,real_age_num)
    age_diff_avg = zerodivide(age_diff_avg, real_age_num)

    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(save_path+'variace_all.txt', 'a+') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(plot_title)
        print(age_diff_avg)
        #print('variance of predicted mean age difference:', np.var(age_diff_avg))
        print('variance of predicted age difference:', np.var(age_diff_vec))
        len_vec = len(age_diff_vec)

        mse_ave_temp = np.matmul(np.matrix(age_diff_vec), np.transpose(np.matrix(age_diff_vec)))/len_vec
        mse_ave = mse_ave_temp[0,0]
        print('MSE:', mse_ave)
        print('age range:', age_range)
        KL_prob_real = kl_count(real_age_vec, age_range)
        #print('whole predict age:', whole_predict_age)
        #print(age_round_f(whole_predict_age, age_range))
        KL_prob_predict = kl_count(age_round_f(whole_predict_age, age_range), age_range)
        KL_value = KL_divergence(KL_prob_real,KL_prob_predict)
        print('KL divergence:', KL_value)
        class_num_vec, sample_num_vec, agediff_vec, agediff_withsign_vec, mse_vec, kl_vec = \
            compute_measures_by_rank(real_age_vec, whole_predict_age, age_range)

        print('class number used', class_num_vec)
        print('sample number used', sample_num_vec)
        print('predicted absolute age difference', agediff_vec)
        print('predicted age difference', agediff_withsign_vec)
        print('mean square error', mse_vec)
        print('Kl divergence', kl_vec)
        # Compute range accuracy for resampling method

        print('non-empty age group:', label_adjust)
        print('standard deviation:', standard_list)
        print('variance:', variance_list)
        print('mean PAD:', mean_list)
        print('median of PAD:', median_list)
        print('maximum of PAD:', maximum_list)
        print('minimum of PAD:', minimum_list)

        sys.stdout = original_stdout  # Reset the standard

    # absolute age difference
    #print(class_num_vec)
    class_num_vec = [int(x) for x in class_num_vec]
    class_num_labels = [str(int(x)) for x in class_num_vec]
    '''
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(class_num_vec, agediff_vec, color='green', align='center', width=0.5)
    plt.xlabel("age number used")
    plt.ylabel("mean absolute age difference")
    #plt.title("mean absolute age difference with different number of ages")
    plt.xticks(class_num_vec, class_num_labels)
    plt.savefig(save_path + 'class_mean_absolute_age_difference.png')
    '''

    # age difference
    class_num_labels = [str(x) for x in class_num_vec]
    '''
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(class_num_vec, agediff_withsign_vec, color='blue', align='center', width=0.5)
    plt.xlabel("age number used")
    plt.ylabel("mean age difference")
    #plt.title("mean age difference with different number of ages")
    plt.xticks(class_num_vec, class_num_labels)
    plt.savefig(save_path + 'class_mean_age_difference.png')
    '''
    # mean square error
    '''
    class_num_labels = [str(x) for x in class_num_vec]
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(class_num_vec, mse_vec, color='red', align='center', width=0.5)
    plt.xlabel("age number used")
    plt.ylabel("mean square error")
    #plt.title("mean square error with different number of ages")
    plt.xticks(class_num_vec, class_num_labels)
    plt.savefig(save_path + 'class_mean_square_error.png')
    

    # mean square error
    class_num_labels = [str(x) for x in class_num_vec]
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(class_num_vec, kl_vec, color='orange', align='center', width=0.5)
    plt.xlabel("age number used")
    plt.ylabel("KL divergence")
    #plt.title("KL divergence with different number of ages")
    plt.xticks(class_num_vec, class_num_labels)
    plt.savefig(save_path + 'class_kl_divergence.png')
    '''

    x_pos = [pos for pos in range(0, bin_range[1]-bin_range[0])]

    '''
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(x_pos, real_age_avg, color='green', align='center', width=0.5)
    plt.xlabel("age group")
    plt.ylabel("mean chronological age")
    #plt.title("chronological age of different age groups")
    plt.xticks(x_pos, labels)
    plt.savefig(save_path + 'real_age_all.png')
    #plt.show()
    '''
    '''
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(x_pos, whole_predict_age_avg, color='blue', align='center', width=0.5)
    plt.xlabel("age group")
    plt.ylabel("predicted mean age")
    #plt.title("predicted age of different age groups")
    plt.xticks(x_pos, labels)
    plt.savefig(save_path + 'predict_age_all.png')
    #plt.show()
    '''

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    #print(labels)
    x_pos_new, age_diff_avg_new, labels_new = reprocess_before_plot(x_pos, age_diff_avg, labels)
    labels_pcc = [int(la) for la in labels_new]
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    plt.bar(x_pos_new, age_diff_avg_new, color=(0/255,149/255,182/255), align='center', width=0.9, alpha=1, edgecolor='dimgray') # 0，47， 167

    poly_d1 = np.polyfit(np.array(x_pos_new), np.array(age_diff_avg_new), 1)
    vec_d1 = np.polyval(poly_d1, x_pos_new)
    plt.plot(x_pos_new, vec_d1, '--', color=(0 / 255, 0 / 255, 0 / 255), label='Linear', linewidth=3)

    poly_d3 = np.polyfit(np.array(x_pos_new), np.array(age_diff_avg_new), 3)
    vec_d3 = np.polyval(poly_d3, x_pos_new)
    plt.plot(x_pos_new, vec_d3, '-', color= (143/255, 75/255, 40/255), label = 'Cubic', linewidth=3)
    poly_d4 = np.polyfit(np.array(x_pos_new), np.array(age_diff_avg_new), 5)
    vec_d4 = np.polyval(poly_d4, x_pos_new)
    plt.plot(x_pos_new, vec_d4, '-.', color=(0/255,47/255,167/255), label = 'Quintic', linewidth=3, markersize=10)

    plt.subplots_adjust(bottom=0.15, left=0.20) #plt.subplots_adjust() #

    if bar_lim != None:
        plt.ylim(bar_lim[0], bar_lim[1])

    #print(x_pos)
    #print(age_diff_avg)
    if if_bar_xlabel:
        plt.xlabel("Chronological age")
    if if_bar_ylabel:
        plt.ylabel("Mean PAD")
    #plt.title("predicted age difference of age groups")
    for i in range(len(labels_new)):
        if i % 8 != 0:
            labels_new[i] = ''
    plt.xticks(x_pos_new, labels_new)
    #plt.tight_layout()
    ##plt.legend(loc='best')
    #plt.ylim((-10, 10))
    plt.savefig(save_path + 'age_diff_all.svg', format='svg', transparent=True)

    # compute pearson

    #print(labels_pcc)
    x_pcc_input = labels_pcc
    #print(x_pcc_input)
    y_pcc_input = age_diff_avg_new
    pcc_mean = scipy.stats.pearsonr(x_pcc_input, y_pcc_input)
    srcc_mean = scipy.stats.spearmanr(x_pcc_input, y_pcc_input)


    # variance
    '''
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(x_pos_adjust, variance_list, color='blue', align='center', width=0.5)
    plt.xlabel("age group")
    plt.ylabel("mean chronological age")
    #plt.title("variance of predicted age")
    plt.xticks(x_pos_adjust, label_adjust)
    plt.savefig(save_path + 'variance_predicted_age.png')
    

    # statistics of PAD
    
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.plot(x_pos_adjust, median_list, 'bs-', label='median', linewidth=2)
    plt.plot(x_pos_adjust, mean_list, 'r*-', label='mean', linewidth=2)
    plt.plot(x_pos_adjust, maximum_list, 'k^-', label='maximum', linewidth=2)
    plt.plot(x_pos_adjust, minimum_list, 'gv-', label='minimum', linewidth=2)
    plt.xlabel("age group")
    plt.ylabel("statistics of PAD")
    #plt.title("statistics of PAD")
    plt.xticks(x_pos_adjust, label_adjust)
    plt.legend()
    plt.savefig(save_path + 'statistics_PAD.png')
    #plt.show()
    '''

    # scatter and linear regression
    age_range_list = [int(x) for x in range(bin_range[0], bin_range[1])]
    age_range_len = len(age_range_list)
    data_len = np.shape(real_age_vec)[0]
    #np.reshape(real_age_vec, (data_len, 1))
    #print(np.shape(real_age_vec), np.shape(age_diff_vec))
    reg = LinearRegression().fit(np.reshape(real_age_vec, (data_len, 1)), np.reshape(age_diff_vec, (data_len, 1)))
    a = reg.coef_
    b = reg.intercept_
    age_linear_vec = np.reshape(np.array(age_range_list) * a + b, (age_range_len, ))

    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(save_path + 'variace_all.txt', 'a+') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print('age-PAD linear regression coefficient and bias:', a, b)
        sys.stdout = original_stdout  # Reset the standard

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.scatter(real_age_vec, age_diff_vec, color = (178/255, 0/255, 32/255), alpha=1, s=30)

    age_min = int(min(real_age_vec)) - 2
    age_max = int(max(real_age_vec)) + 2
    age_range_list_scatter = list(range(age_min, age_max+1))

    poly_d1 = np.polyfit(np.array(real_age_vec), np.array(age_diff_vec), 1)
    vec_d1 = np.polyval(poly_d1, np.array(age_range_list_scatter))
    plt.plot(age_range_list_scatter, vec_d1, '--', color=(0 / 255, 0 / 255, 0 / 255), label='Linear', linewidth=3)

    poly_d3 = np.polyfit(np.array(real_age_vec), np.array(age_diff_vec), 3)
    vec_d3 = np.polyval(poly_d3, np.array(age_range_list_scatter))
    plt.plot(age_range_list_scatter, vec_d3, '-', color=(143/255, 75/255, 40/255), label='Cubic', linewidth=3)
    poly_d5 = np.polyfit(np.array(real_age_vec), np.array(age_diff_vec), 5)
    vec_d5 = np.polyval(poly_d5, np.array(age_range_list_scatter))
    plt.plot(age_range_list_scatter, vec_d5, '-.', color=(0/255,47/255,167/255), label='Quintic', linewidth=3, markersize=10)
    plt.subplots_adjust(bottom=0.15, left=0.20) #
    #plt.plot(age_range_list, age_linear_vec, color= (76/255, 0, 9/255), linewidth = 3)
    #plt.title("PAD and chronological age")
    '''
    if if_scatter_xlabel:
        
    '''
    if if_scatter_xlabel:
        plt.xlabel("Chronological age")
    if if_scatter_ylabel:
        plt.ylabel("PAD")
    #plt.tight_layout()
    #plt.ylim((-20, 20))
    plt.savefig(save_path + 'PAD_scatter.svg', format='svg', transparent=True)


    #----------------------------------------------------
    ## Normalized PAD
    #
    real_age_vec_zscore, age_diff_vec_zscore, _ = zscore_PAD(real_age_vec, age_diff_vec)
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.scatter(real_age_vec_zscore, age_diff_vec_zscore, color=(178 / 255, 0 / 255, 32 / 255), alpha=1, s=30)

    age_min = int(min(real_age_vec_zscore)) - 2
    age_max = int(max(real_age_vec_zscore)) + 2
    age_range_list_scatter = list(range(age_min, age_max + 1))
    #print(age_range_list_scatter)

    poly_d1 = np.polyfit(np.array(real_age_vec_zscore), np.array(age_diff_vec_zscore), 1)
    vec_d1 = np.polyval(poly_d1, np.array(age_range_list_scatter))
    plt.plot(age_range_list_scatter, vec_d1, '--', color=(0 / 255, 0 / 255, 0 / 255), label='Linear', linewidth=3)

    poly_d3 = np.polyfit(np.array(real_age_vec_zscore), np.array(age_diff_vec_zscore), 3)
    vec_d3 = np.polyval(poly_d3, np.array(age_range_list_scatter))
    plt.plot(age_range_list_scatter, vec_d3, '-', color=(143 / 255, 75 / 255, 40 / 255), label='Cubic', linewidth=3)
    poly_d5 = np.polyfit(np.array(real_age_vec_zscore), np.array(age_diff_vec_zscore), 5)
    vec_d5 = np.polyval(poly_d5, np.array(age_range_list_scatter))
    plt.plot(age_range_list_scatter, vec_d5, '-.', color=(0 / 255, 47 / 255, 167 / 255), label='Quintic', linewidth=3,
             markersize=10)
    plt.subplots_adjust(bottom=0.15, left=0.20)  #
    # plt.plot(age_range_list, age_linear_vec, color= (76/255, 0, 9/255), linewidth = 3)
    # plt.title("PAD and chronological age")
    '''
    if if_scatter_xlabel:

    '''
    if if_scatter_xlabel:
        plt.xlabel("Chronological age")
    if if_scatter_ylabel:
        plt.ylabel("Normalized PAD")
    # plt.tight_layout()
    # plt.ylim((-20, 20))
    plt.savefig(save_path + 'PAD_scatter_normalized.svg', format='svg', transparent=True)


    real_age_zscore_list, age_diff_vec_zscore_avg = compute_avg(real_age_vec_zscore, age_diff_vec_zscore)
    x_pcc_input_zscore_mean = real_age_zscore_list
    y_pcc_input_zscore_mean = age_diff_vec_zscore_avg
    pcc_mean_zscore = scipy.stats.pearsonr(x_pcc_input_zscore_mean, y_pcc_input_zscore_mean)
    srcc_mean_zscore = scipy.stats.spearmanr(x_pcc_input_zscore_mean, y_pcc_input_zscore_mean)


    #plt.show()
    # compute pcc
    x_pcc_input = real_age_vec
    y_pcc_input = age_diff_vec
    pcc_all = scipy.stats.pearsonr(x_pcc_input, y_pcc_input)
    srcc_all = scipy.stats.spearmanr(x_pcc_input, y_pcc_input)

    # Normalized PAD
    x_pcc_input_zscore = real_age_vec_zscore
    y_pcc_input_zscore = age_diff_vec_zscore
    pcc_all_zscore = scipy.stats.pearsonr(x_pcc_input_zscore, y_pcc_input_zscore)
    srcc_all_zscore = scipy.stats.spearmanr(x_pcc_input_zscore, y_pcc_input_zscore)


    #APAD scatter and linear regression
    age_range_list = [int(x) for x in range(bin_range[0], bin_range[1])]
    ave_predict_age_dict = dict(zip(age_range_list, whole_predict_age_avg))
    predict_age_vec_apad = np.zeros(np.shape(real_age_vec))

    for k in range(len(real_age_vec)):
        predict_age_vec_apad[k] = ave_predict_age_dict[int(real_age_vec[k])]

    age_range_len = len(age_range_list)
    data_len = np.shape(real_age_vec)[0]
    # np.reshape(real_age_vec, (data_len, 1))
    # print(np.shape(real_age_vec), np.shape(age_diff_vec))
    reg = LinearRegression().fit(np.reshape(real_age_vec, (data_len, 1)), np.reshape(whole_predict_age - predict_age_vec_apad, (data_len, 1)))
    a = reg.coef_
    b = reg.intercept_
    age_linear_vec = np.reshape(np.array(age_range_list) * a + b, (age_range_len,))

    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(save_path + 'variace_all.txt', 'a+') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print('age-APAD linear regression coefficient and bias:', a, b)
        sys.stdout = original_stdout  # Reset the standard

    '''
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.scatter(real_age_vec, whole_predict_age - predict_age_vec_apad, alpha=0.5, s=5)
    plt.plot(age_range_list, age_linear_vec)
    #plt.title("APAD and chronological age")
    plt.xlabel("chronological age")
    plt.ylabel("APAD")
    plt.savefig(save_path + 'APAD_scatter.png')
    # plt.show()
    '''

    return [pcc_mean, pcc_all, srcc_mean, srcc_all], [pcc_mean_zscore, pcc_all_zscore, srcc_mean_zscore, srcc_all_zscore]


def compute_error(whole_path, plot_title, save_path, bin_range = [38, 86]):
    #plt.style.use('ggplot')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    whole_pd = pd.read_csv(whole_path)
    real_age = whole_pd['real age'].values
    whole_predict_age = whole_pd['predict age'].values
    age_diff_vec = whole_pd['age difference'].values
    age_group_dict = {}
    #age_diff_vec = whole_pd['age difference'].values
    if bin_range == [38, 86]:
        labels = ['38-50', '51-65', '66-70', '71-75', '76-86']
        group_list = [[38, 50], [51, 65], [66, 70], [71, 75], [76, 86]]
    elif bin_range == [42, 97]:
        labels = ['42-50', '51-65', '66-70', '71-80', '81-97']
        group_list = [[42, 50], [51, 65], [66, 70], [71, 80], [81, 97]]
    elif bin_range == [6, 65]:
        labels = ['6-15', '16-25', '26-35', '36-45', '46-65']
        group_list = [[6, 15], [16, 25], [26, 35], [36, 45], [46, 65]]

    for k in range(5):
        age_range = group_list[k]
        for j in range(age_range[0], age_range[1]+1):
            age_group_dict[str(j)] = k

    real_age_num = np.zeros(5)
    real_age_avg = np.zeros(5)
    whole_predict_age_avg = np.zeros(5)
    age_diff_avg = np.zeros(5)

    for i in range(len(whole_predict_age)):
        user_id,predict_age,real_age,age_diff = whole_pd.iloc[i]
        #print(user_id,predict_age,real_age,age_diff)
        group_id = age_group_dict[str(int(real_age))]

        whole_predict_age_avg[group_id] += predict_age

        real_age_avg[group_id] += real_age
        real_age_num[group_id] += 1

        age_diff_avg[group_id] += age_diff


    real_age_avg = np.true_divide(real_age_avg,real_age_num)
    whole_predict_age_avg = np.true_divide(whole_predict_age_avg,real_age_num)
    age_diff_avg = np.true_divide(age_diff_avg, real_age_num)


    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(save_path+'variace.txt', 'a+') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(plot_title)
        print(age_diff_avg)
        print('variance of predicted mean age difference:', np.var(age_diff_avg))
        print('variance of predicted age difference:', np.var(age_diff_vec))
        sys.stdout = original_stdout  # Reset the standard



    x_pos = [pos for pos in range(0, 5)]

    '''
    plt.figure()
    plt.bar(x_pos, real_age_avg, color='green')
    plt.xlabel("age group")
    plt.ylabel("chronological mean age")
    plt.title("chronological age of different age groups")
    plt.xticks(x_pos, labels)
    plt.savefig(save_path + 'real_age_group.png')
    #plt.show()

    plt.figure()
    plt.bar(x_pos, whole_predict_age_avg, color='blue')
    plt.xlabel("age group")
    plt.ylabel("mean predicted  age")
    #plt.title("predicted age of different age groups")
    plt.xticks(x_pos, labels)
    plt.savefig(save_path + 'predict_age_group.png')
    #plt.show()

    plt.figure()
    plt.bar(x_pos, age_diff_avg, color='red')
    plt.xlabel("age group")
    plt.ylabel("mean predict age difference")
    #plt.title("predicted age difference of age groups")
    plt.xticks(x_pos, labels)
    plt.savefig(save_path + 'age_diff_group.png')
    #plt.show()
    '''



def age_statistics_real(whole_path, save_path, bin_range = [38, 86]):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    whole_pd = pd.read_csv(whole_path)
    real_age = whole_pd['real age'].values
    plt.figure()
    plt.hist(real_age, bins = range(bin_range[0], bin_range[1]), color='blue')
    #plt.title("real age distribution")
    plt.xlabel("age")
    plt.ylabel("number")
    plt.savefig(save_path + 'real_age.png')
    #plt.show()

def age_statistics(whole_path, plot_title, save_path, bin_range = [38, 86]):
    whole_pd = pd.read_csv(whole_path)
    whole_predict_age = whole_pd['predict age'].values
    real_age = whole_pd['real age'].values
    plt.figure()
    plt.hist(whole_predict_age, bins = range(bin_range[0], bin_range[1]), color='green')
    #plt.title(plot_title + " age distribution")
    plt.xlabel("age")
    plt.ylabel("number")
    plt.savefig(save_path  + plot_title + '.png')
    #plt.show()


def age_scatter(left_predict_age, right_predict_age, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.scatter(left_predict_age, right_predict_age, alpha=0.5, s=5)
    plt.plot(left_predict_age, left_predict_age)
    plt.title("left and right predicted age")
    plt.xlabel("left hemisphere")
    plt.ylabel("right hemisphere")
    plt.savefig(save_path)
    plt.show()



def plot_whole_brain(whole_path, plot_title, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    whole_pd = pd.read_csv(whole_path)
    whole_predict_age = whole_pd['predict age'].values
    real_age = whole_pd['real age'].values

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    len_vec = range(1, len(whole_predict_age)+1)


    plt.scatter(len_vec, whole_predict_age, c='b', alpha=0.5, label='right hemisphere',s=4)
    #plt.scatter(len_vec, left_predict_age, c='r', alpha=0.5, label='left hemisphere',s=4)
    plt.scatter(len_vec, real_age, c='g', alpha=0.5, label='chronological',s=4)
    #plt.title(plot_title+' whole brain age prediction')
    plt.xlabel("subjects")
    plt.ylabel("age")
    plt.legend(loc='upper left')
    plt.savefig(save_path + 'whole.png')
    plt.show()



def plot_left_right(left_path, right_path, plot_title):
    left_pd = pd.read_csv(left_path)
    left_predict_age = left_pd['predict age'].values
    #print(left_predict_age.shape)

    right_pd = pd.read_csv(right_path)
    right_predict_age = right_pd['predict age'].values

    real_age = right_pd['real age'].values

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    len_vec = range(1, len(left_predict_age)+1)


    plt.scatter(len_vec, right_predict_age, c='b', alpha=0.5, label='right hemisphere',s=4)
    plt.scatter(len_vec, left_predict_age, c='r', alpha=0.5, label='left hemisphere',s=4)
    plt.scatter(len_vec, real_age, c='g', alpha=0.5, label='chronological',s=4)
    plt.title(plot_title+" left and right predicted age")
    plt.xlabel("subjects")
    plt.ylabel("age")
    plt.legend(loc='upper left')
    plt.savefig(left_path + '_left_right_plot.png')
    plt.show()


    age_scatter(left_predict_age, right_predict_age, save_path=left_path+'_left_right_scatter.png')







def plot_sequence(whole_path, plot_title, save_path, bin_range = [38, 86], if_bar_xlabel = False, if_bar_ylabel = False, if_scatter_xlabel = False, if_scatter_ylabel = False, bar_lim = None):
    age_statistics_real(whole_path, save_path, bin_range)
    age_statistics(whole_path, plot_title, save_path, bin_range)
    compute_error(whole_path, plot_title, save_path, bin_range)
    corr_list = compute_error_all(whole_path, plot_title, save_path,bin_range, if_bar_xlabel, if_bar_ylabel, if_scatter_xlabel, if_scatter_ylabel, bar_lim)
    compute_class_accuracy(whole_path, plot_title, save_path, bin_range)
    return corr_list


def bar_mean_all(corr_list, corr_list_linear, corr_list_quadratic, corr_list_zscore, corr_list_linear_zscore, corr_list_quadratic_zscore,  save_path, if_bar_xlabel=True, if_bar_ylabel=True):

    if_legend = True

    # PCC
    pcc_mean = format(abs(corr_list[0][0])+0.01, '.2f')
    pcc_mean_linear = format(abs(corr_list_linear[0][0])+0.01, '.2f')
    pcc_mean_quadratic = format(abs(corr_list_quadratic[0][0])+0.01, '.2f')
    pcc_all = format(abs(corr_list[1][0])+0.01, '.2f')
    pcc_all_linear = format(abs(corr_list_linear[1][0])+0.01, '.2f')
    pcc_all_quadratic = format(abs(corr_list_quadratic[1][0])+0.01, '.2f')

    score_mean = [pcc_mean, pcc_mean_linear, pcc_mean_quadratic]
    score_all = [pcc_all, pcc_all_linear, pcc_all_quadratic]

    df = pd.DataFrame([['Mean PAD','None',pcc_mean],['Mean PAD','Linear',pcc_mean_linear],['Mean PAD','Quadratic',pcc_mean_quadratic],
                        ['PAD','None',pcc_all], ['PAD','Linear',pcc_all_linear],['PAD','Quadratic',pcc_all_quadratic]],columns=['Type','Method','PCC'])

    index = ['None', 'Linear', 'Quadratic']
    df = pd.DataFrame({'PAD': score_all, 'Mean PAD': score_mean}, index=index).astype(float)


    ax = df.plot.bar(rot=0, color={"Mean PAD": (0/255,149/255,182/255), 'PAD': (178/255, 0/255, 32/255)},
                     figsize = (img_width, img_height), width=0.88,  edgecolor = 'dimgray', legend=if_legend, bottom=-0.01, ylim = -0.01) #legend=True,

    #plt.gca().invert_yaxis()


    for container in ax.containers:
        #print(container)
        ax.bar_label(container)



    fig = ax.get_figure()
    #plt.tight_layout()
    fig.savefig(save_path + 'bar_mean_all_pcc.svg', format='svg', transparent=True)

    # SRCC
    srcc_mean = format(abs(corr_list[2][0]), '.2f')
    srcc_mean_linear = format(abs(corr_list_linear[2][0]), '.2f')
    srcc_mean_quadratic = format(abs(corr_list_quadratic[2][0]), '.2f')
    srcc_all = format(abs(corr_list[3][0]), '.2f')
    srcc_all_linear = format(abs(corr_list_linear[3][0]), '.2f')
    srcc_all_quadratic = format(abs(corr_list_quadratic[3][0]), '.2f')

    score_mean = [srcc_mean, srcc_mean_linear, srcc_mean_quadratic]
    score_all = [srcc_all, srcc_all_linear, srcc_all_quadratic]

    df = pd.DataFrame([['Mean PAD', 'None', srcc_mean], ['Mean PAD', 'Linear', srcc_mean_linear],
                       ['Mean PAD', 'Quadratic', srcc_mean_quadratic],
                       ['PAD', 'None', srcc_all], ['PAD', 'Linear', srcc_all_linear],
                       ['PAD', 'Quadratic', srcc_all_quadratic]], columns=['Type', 'Method', 'PCC'])

    index = ['None', 'Linear', 'Quadratic']
    df = pd.DataFrame({'PAD': score_all, 'Mean PAD': score_mean}, index=index).astype(float)

    ax = df.plot.bar(rot=0, color={"Mean PAD": (0 / 255, 149 / 255, 182 / 255), 'PAD': (178/255, 0/255, 32/255)},
                     figsize=(img_width, img_height), width=0.88,  edgecolor = 'dimgray', legend=if_legend,bottom=-0.01, ylim = -0.01) #legend=True,


    for container in ax.containers:
        ax.bar_label(container)


    fig = ax.get_figure()
    #plt.tight_layout()
    fig.savefig(save_path + 'bar_mean_all_srcc.svg', format='svg', transparent=True)


#----------------------------------
    # Zscore PCC
    pcc_mean = format(abs(corr_list_zscore[0][0]) + 0.01, '.2f')
    pcc_mean_linear = format(abs(corr_list_linear_zscore[0][0]) + 0.01, '.2f')
    pcc_mean_quadratic = format(abs(corr_list_quadratic_zscore[0][0]) + 0.01, '.2f')
    pcc_all = format(abs(corr_list_zscore[1][0]) + 0.01, '.2f')
    pcc_all_linear = format(abs(corr_list_linear_zscore[1][0]) + 0.01, '.2f')
    pcc_all_quadratic = format(abs(corr_list_quadratic_zscore[1][0]) + 0.01, '.2f')

    score_mean = [pcc_mean, pcc_mean_linear, pcc_mean_quadratic]
    score_all = [pcc_all, pcc_all_linear, pcc_all_quadratic]

    df = pd.DataFrame([['Mean PAD', 'None', pcc_mean], ['Mean PAD', 'Linear', pcc_mean_linear],
                       ['Mean PAD', 'Quadratic', pcc_mean_quadratic],
                       ['PAD', 'None', pcc_all], ['PAD', 'Linear', pcc_all_linear],
                       ['PAD', 'Quadratic', pcc_all_quadratic]], columns=['Type', 'Method', 'PCC'])

    index = ['None', 'Linear', 'Quadratic']
    df = pd.DataFrame({'PAD': score_all, 'Mean PAD': score_mean}, index=index).astype(float)

    ax = df.plot.bar(rot=0, color={"Mean PAD": (0 / 255, 149 / 255, 182 / 255), 'PAD': (178 / 255, 0 / 255, 32 / 255)},
                     figsize=(img_width, img_height), width=0.88, edgecolor='dimgray', legend=if_legend, bottom=-0.01,
                     ylim=[-0.01, 1])  # legend=True,

    # plt.gca().invert_yaxis()

    for container in ax.containers:
        # print(container)
        ax.bar_label(container)

    fig = ax.get_figure()
    # plt.tight_layout()
    fig.savefig(save_path + 'zscore_bar_mean_all_pcc.svg', format='svg', transparent=True)

    # SRCC
    srcc_mean = format(abs(corr_list_zscore[2][0]), '.2f')
    srcc_mean_linear = format(abs(corr_list_linear_zscore[2][0]), '.2f')
    srcc_mean_quadratic = format(abs(corr_list_quadratic_zscore[2][0]), '.2f')
    srcc_all = format(abs(corr_list_zscore[3][0]), '.2f')
    srcc_all_linear = format(abs(corr_list_linear_zscore[3][0]), '.2f')
    srcc_all_quadratic = format(abs(corr_list_quadratic_zscore[3][0]), '.2f')

    score_mean = [srcc_mean, srcc_mean_linear, srcc_mean_quadratic]
    score_all = [srcc_all, srcc_all_linear, srcc_all_quadratic]

    df = pd.DataFrame([['Mean PAD', 'None', srcc_mean], ['Mean PAD', 'Linear', srcc_mean_linear],
                       ['Mean PAD', 'Quadratic', srcc_mean_quadratic],
                       ['PAD', 'None', srcc_all], ['PAD', 'Linear', srcc_all_linear],
                       ['PAD', 'Quadratic', srcc_all_quadratic]], columns=['Type', 'Method', 'PCC'])

    index = ['None', 'Linear', 'Quadratic']
    df = pd.DataFrame({'PAD': score_all, 'Mean PAD': score_mean}, index=index).astype(float)

    ax = df.plot.bar(rot=0, color={"Mean PAD": (0 / 255, 149 / 255, 182 / 255), 'PAD': (178 / 255, 0 / 255, 32 / 255)},
                     figsize=(img_width, img_height), width=0.88, edgecolor='dimgray', legend=if_legend, bottom=-0.01,
                     ylim=[-0.01, 1])  # legend=True,

    for container in ax.containers:
        ax.bar_label(container)

    fig = ax.get_figure()
    # plt.tight_layout()
    fig.savefig(save_path + 'zscore_bar_mean_all_srcc.svg', format='svg', transparent=True)




def compute_error_all_corrected(whole_path, plot_title, save_path, dataset_name = 'ukbiobank', ifcorrection = True):
    #plt.show()plt.style.use('ggplot')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    whole_pd = pd.read_csv(whole_path)
    real_age_vec = whole_pd['real age'].values
    path_name_add = ''
    if ifcorrection == True:
        whole_predict_age = whole_pd['corrected predict age'].values
        age_diff_vec = whole_predict_age - real_age_vec
        path_name_add = 'Corrected '
    else:
        whole_predict_age = whole_pd['predict age'].values
        age_diff_vec = whole_predict_age - real_age_vec


    if dataset_name == 'ukbiobank':
        age_range = range(38, 86)
        labels = [str(int(x)) for x in range(38, 86)]

    class_num = len(age_range)
    age_group_dict = {}
    # group_list = [[38, 50], [51, 65], [66, 70], [71, 75], [76, 86]]
    for k in range(class_num):
        age_group_dict[labels[k]] = k

    # print(age_group_dict)
    real_age_num = np.zeros(class_num)
    real_age_avg = np.zeros(class_num)
    whole_predict_age_avg = np.zeros(class_num)
    age_diff_avg = np.zeros(class_num)

    age_list_dict = {}
    for k in range(class_num):
        age_list_dict[labels[k]] = []
    agediff_list_dict = {}
    for k in range(class_num):
        agediff_list_dict[labels[k]] = []

    # used to compute whole standard deviation
    agediff_list_stddev_dict = {}
    age_list_stddev_dict = {}
    for k in range(class_num):
        age_list_stddev_dict[labels[k]] = []
        agediff_list_stddev_dict[labels[k]] = []

    pick_sample_num = 20

    for i in range(len(whole_predict_age)):
        if ifcorrection == True:
            user_id, _, real_age, _, predict_age, age_diff = whole_pd.iloc[i]
        else:
            user_id, predict_age, real_age, age_diff = whole_pd.iloc[i, 0:4]
        # print(user_id,predict_age,real_age,age_diff)
        group_id = age_group_dict[str(int(real_age))]

        age_list_stddev_dict[str(int(real_age))].append(predict_age)
        if len(age_list_dict[str(int(real_age))]) <= pick_sample_num:
            age_list_dict[str(int(real_age))].append(predict_age)
        whole_predict_age_avg[group_id] += predict_age

        real_age_avg[group_id] += real_age
        real_age_num[group_id] += 1

        age_diff_avg[group_id] += predict_age - real_age

        agediff_list_stddev_dict[str(int(real_age))].append(predict_age - real_age)

        # age diff list for computing median, maximum and minimum
        if len(agediff_list_dict[str(int(real_age))]) <= pick_sample_num:
            agediff_list_dict[str(int(real_age))].append(predict_age - real_age)
    # print(age_list_dict)

    # compute stardard
    standard_list = []
    variance_list = []
    mean_list = []
    median_list = []
    maximum_list = []
    minimum_list = []
    x_pos_adjust = []
    label_adjust = []
    for k in range(class_num):
        temp_age_list = age_list_dict[labels[k]]
        if len(temp_age_list) >= pick_sample_num:
            standard_list.append(np.std(temp_age_list))
            variance_list.append(np.var(temp_age_list))

        temp_agediff_list = agediff_list_dict[labels[k]]
        if len(temp_agediff_list) >= pick_sample_num:  # temp_agediff_list != []:
            x_pos_adjust.append(k)
            label_adjust.append(labels[k])
            mean_list.append(np.mean(temp_agediff_list))
            median_list.append(statistics.median(temp_agediff_list))
            maximum_list.append(max(temp_agediff_list))
            minimum_list.append(min(temp_agediff_list))

    standard_list = []
    std_x_pos_adjust = []
    std_label_adjust = []
    std_pick_sample_num = 5
    for k in range(class_num):
        temp_age_list = age_list_stddev_dict[labels[k]]
        if len(temp_age_list) >= std_pick_sample_num:
            standard_list.append(np.std(temp_age_list))

        temp_agediff_list = agediff_list_stddev_dict[labels[k]]
        if len(temp_agediff_list) >= pick_sample_num:  # temp_agediff_list != []:
            std_x_pos_adjust.append(k)
            std_label_adjust.append(labels[k])

    real_age_avg = zerodivide(real_age_avg, real_age_num)
    whole_predict_age_avg = zerodivide(whole_predict_age_avg, real_age_num)
    age_diff_avg = zerodivide(age_diff_avg, real_age_num)

    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(save_path + path_name_add +'variace_all.txt', 'a+') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(plot_title)
        print(age_diff_avg)
        # print('variance of predicted mean age difference:', np.var(age_diff_avg))
        print('variance of predicted age difference:', np.var(age_diff_vec))
        len_vec = len(age_diff_vec)

        mse_ave_temp = np.matmul(np.matrix(age_diff_vec), np.transpose(np.matrix(age_diff_vec))) / len_vec
        mse_ave = mse_ave_temp[0, 0]
        print('MSE:', mse_ave)
        KL_prob_real = kl_count(real_age_vec, age_range)
        KL_prob_predict = kl_count(age_round_f(whole_predict_age, age_range), age_range)
        KL_value = KL_divergence(KL_prob_real, KL_prob_predict)
        print('KL divergence:', KL_value)
        class_num_vec, sample_num_vec, agediff_vec, agediff_withsign_vec, mse_vec, kl_vec = \
            compute_measures_by_rank(real_age_vec, whole_predict_age, age_range)

        print('class number used', class_num_vec)
        print('sample number used', sample_num_vec)
        print('predicted absolute age difference', agediff_vec)
        print('predicted age difference', agediff_withsign_vec)
        print('mean square error', mse_vec)
        print('Kl divergence', kl_vec)
        # Compute range accuracy for resampling method

        print('non-empty age group:', label_adjust)
        print('standard deviation:', standard_list)
        print('variance:', variance_list)
        print('mean PAD:', mean_list)
        print('median of PAD:', median_list)
        print('maximum of PAD:', maximum_list)
        print('minimum of PAD:', minimum_list)

        sys.stdout = original_stdout  # Reset the standard

    # absolute age difference
    # print(class_num_vec)
    class_num_vec = [int(x) for x in class_num_vec]
    class_num_labels = [str(int(x)) for x in class_num_vec]
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(class_num_vec, agediff_vec, color='green', align='center', width=0.5)
    plt.xlabel("Age number used")
    plt.ylabel("mean absolute age difference")
    plt.title(path_name_add+"mean absolute age difference with different number of ages")
    plt.xticks(class_num_vec, class_num_labels)
    plt.savefig(save_path + path_name_add + 'corrected_class_mean_absolute_age_difference.png')

    # age difference
    class_num_labels = [str(x) for x in class_num_vec]
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(class_num_vec, agediff_withsign_vec, color='blue', align='center', width=0.5)
    plt.xlabel("Age number used")
    plt.ylabel("mean age difference")
    plt.title(path_name_add + "mean age difference with different number of ages")
    plt.xticks(class_num_vec, class_num_labels)
    plt.savefig(save_path + path_name_add + 'class_mean_age_difference.png')

    # mean square error
    class_num_labels = [str(x) for x in class_num_vec]
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(class_num_vec, mse_vec, color='red', align='center', width=0.5)
    plt.xlabel("Age number used")
    plt.ylabel("mean square error")
    plt.title(path_name_add +"mean square error with different number of ages")
    plt.xticks(class_num_vec, class_num_labels)
    plt.savefig(save_path + path_name_add + 'class_mean_square_error.png')

    # mean square error
    class_num_labels = [str(x) for x in class_num_vec]
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(class_num_vec, kl_vec, color='orange', align='center', width=0.5)
    plt.xlabel("Age number used")
    plt.ylabel("KL divergence")
    plt.title(path_name_add + "KL divergence with different number of ages")
    plt.xticks(class_num_vec, class_num_labels)
    plt.savefig(save_path + path_name_add + 'class_kl_divergence.png')

    x_pos = [pos for pos in range(0, class_num)]

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(x_pos, real_age_avg, color='green', align='center', width=0.5)
    plt.xlabel("age group")
    plt.ylabel("chronological mean age")
    plt.title(path_name_add + "chronological age of different age groups")
    plt.xticks(x_pos, labels)
    plt.savefig(save_path + path_name_add + 'real_age_all.png')
    # plt.show()

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(x_pos, whole_predict_age_avg, color='blue', align='center', width=0.5)
    plt.xlabel("age group")
    plt.ylabel("predicted mean age")
    plt.title(path_name_add + "predicted age of different age groups")
    plt.xticks(x_pos, labels)
    plt.savefig(save_path + path_name_add + 'predict_age_all.png')
    # plt.show()

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(x_pos, age_diff_avg, color='red', align='center', width=0.1)
    # print(x_pos)
    # print(age_diff_avg)
    plt.xlabel("age group")
    plt.ylabel("predicted mean age difference")
    plt.title(path_name_add + "predicted age difference of age groups")
    plt.xticks(x_pos, labels)
    plt.ylim((-10, 10))
    plt.savefig(save_path + path_name_add + 'age_diff_all.png')

    # variance
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.bar(x_pos_adjust, variance_list, color='blue', align='center', width=0.5)
    plt.xlabel("age group")
    plt.ylabel("chronological mean age")
    plt.title(path_name_add + "variance of predicted age")
    plt.xticks(x_pos_adjust, label_adjust)
    plt.savefig(save_path + path_name_add + 'variance_predicted_age.png')

    # statistics of PAD
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.plot(x_pos_adjust, median_list, 'bs-', label='median', linewidth=2)
    plt.plot(x_pos_adjust, mean_list, 'r*-', label='mean', linewidth=2)
    plt.plot(x_pos_adjust, maximum_list, 'k^-', label='maximum', linewidth=2)
    plt.plot(x_pos_adjust, minimum_list, 'gv-', label='minimum', linewidth=2)
    plt.xlabel("age group")
    plt.ylabel("statistics of PAD")
    plt.title(path_name_add + "statistics of PAD")
    plt.xticks(x_pos_adjust, label_adjust)
    plt.legend()
    plt.savefig(save_path + path_name_add + 'statistics_PAD.png')
    # plt.show()

    # scatter and linear regression
    if dataset_name == 'ukbiobank':
        age_range_list = [int(x) for x in range(38, 86)]
    age_range_len = len(age_range_list)
    data_len = np.shape(real_age_vec)[0]
    # np.reshape(real_age_vec, (data_len, 1))
    # print(np.shape(real_age_vec), np.shape(age_diff_vec))
    reg = LinearRegression().fit(np.reshape(real_age_vec, (data_len, 1)), np.reshape(age_diff_vec, (data_len, 1)))
    a = reg.coef_
    b = reg.intercept_
    age_linear_vec = np.reshape(np.array(age_range_list) * a + b, (age_range_len,))

    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(save_path + path_name_add + 'variace_all.txt', 'a+') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print('age-PAD linear regression coefficient and bias:', a, b)
        sys.stdout = original_stdout  # Reset the standard

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.scatter(real_age_vec, age_diff_vec, alpha=0.5, s=10)
    plt.plot(age_range_list, age_linear_vec, 'k', linewidth = 5)
    plt.title(path_name_add + "PAD and chronological age")
    plt.xlabel("chronological age")
    plt.ylabel("predict age difference")
    plt.ylim((-20, 20))
    plt.savefig(save_path + path_name_add + 'PAD_scatter.png')
    # plt.show()

    # APAD scatter and linear regression
    ave_predict_age_dict = dict(zip(age_range_list, whole_predict_age_avg))
    predict_age_vec_apad = np.zeros(np.shape(real_age_vec))

    for k in range(len(real_age_vec)):
        predict_age_vec_apad[k] = ave_predict_age_dict[real_age_vec[k]]

    age_range_len = len(age_range_list)
    data_len = np.shape(real_age_vec)[0]
    # np.reshape(real_age_vec, (data_len, 1))
    # print(np.shape(real_age_vec), np.shape(age_diff_vec))
    reg = LinearRegression().fit(np.reshape(real_age_vec, (data_len, 1)),
                                 np.reshape(whole_predict_age - predict_age_vec_apad, (data_len, 1)))
    a = reg.coef_
    b = reg.intercept_
    age_linear_vec = np.reshape(np.array(age_range_list) * a + b, (age_range_len,))

    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(save_path + path_name_add + 'variace_all.txt', 'a+') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print('age-APAD linear regression coefficient and bias:', a, b)
        sys.stdout = original_stdout  # Reset the standard

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.scatter(real_age_vec, whole_predict_age - predict_age_vec_apad, alpha=0.5, s=5)
    plt.plot(age_range_list, age_linear_vec)
    plt.title(path_name_add +"APAD and chronological age")
    plt.xlabel("chronological age")
    plt.ylabel("predict age difference")
    plt.savefig(save_path + path_name_add +'APAD_scatter.png')
    # plt.show()

    ## NPAD
    std_x_pos_adjust = []

    std_age_range_list = [int(x) for x in std_label_adjust]
    std_dev_pad_dict = dict(zip(std_age_range_list, standard_list))
    std_predict_mean_age_vec = []
    std_real_age_vec = []
    std_dev_predict_age = []
    std_predict_age_vec = []
    for k in range(len(real_age_vec)):
        if real_age_vec[k] in std_age_range_list:
            std_real_age_vec.append(real_age_vec[k])
            std_predict_mean_age_vec.append(ave_predict_age_dict[real_age_vec[k]])
            std_dev_predict_age.append(std_dev_pad_dict[real_age_vec[k]])
            std_predict_age_vec.append(whole_predict_age[k])

    std_real_age_vec = np.array(std_real_age_vec)
    std_dev_predict_age = np.array(std_dev_predict_age)
    std_predict_mean_age_vec = np.array(std_predict_mean_age_vec)
    std_predict_age_vec = np.array(std_predict_age_vec)

    age_range_len = len(age_range_list)
    data_len = np.shape(std_real_age_vec)[0]
    # np.reshape(real_age_vec, (data_len, 1))
    # print(np.shape(real_age_vec), np.shape(age_diff_vec))
    reg = LinearRegression().fit(np.reshape(std_real_age_vec, (data_len, 1)),
                                 np.reshape((std_predict_age_vec - std_predict_mean_age_vec) / std_dev_predict_age,
                                            (data_len, 1)))
    a = reg.coef_
    b = reg.intercept_
    age_linear_vec = np.reshape(np.array(age_range_list) * a + b, (age_range_len,))
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(save_path + path_name_add + 'variace_all.txt', 'a+') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print('age-NPAD linear regression coefficient and bias:', a, b)
        sys.stdout = original_stdout  # Reset the standard

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    plt.scatter(std_real_age_vec, (std_predict_age_vec - std_predict_mean_age_vec) / std_dev_predict_age, c="g",
                alpha=0.5, s=5)
    plt.plot(age_range_list, age_linear_vec)
    plt.title(path_name_add + "NPAD and chronological age")
    plt.xlabel("chronological age")
    plt.ylabel("NPAD")
    plt.savefig(save_path + path_name_add + 'NPAD_scatter.png')



def plot_sequence_general(whole_path, plot_title, save_path, ifcorrection = True, dataset_name='ukbiobank'):
    age_statistics_real(whole_path, save_path)
    age_statistics(whole_path, plot_title, save_path)
    #compute_error_general(whole_path, plot_title, save_path, ifcorrection = ifcorrection)
    #compute_error_all(whole_path, plot_title, save_path, ifcorrection = ifcorrection)
    if ifcorrection == True:
        compute_error_all_corrected(whole_path, plot_title, save_path, dataset_name=dataset_name, ifcorrection = False)
        compute_error_all_corrected(whole_path, plot_title, save_path, dataset_name=dataset_name, ifcorrection = True)
    else:
        compute_error_all_corrected(whole_path, plot_title, save_path, dataset_name=dataset_name, ifcorrection=False)

    #compute_class_accuracy_general(whole_path, plot_title, save_path, ifcorrection = ifcorrection)


def linear_regression_endpoints_v1(dataset_name, age_real, age_predict, apad, linear_apad, quad_apad,
                                age_pad, age_pad_linear, age_pad_quad, regression_index,
                                regression_index_name, regression_index_display_name, save_path,  method = 'linear'):
    TINY_SIZE = 19#39
    SMALL_SIZE = 22#42
    MEDIUM_SIZE = 26#46
    BIGGER_SIZE = 26#46

    plt.rc('font', size=16)  # 35 controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams["legend.frameon"] = False
    # plt.rc('legend',**{'fontsize':16})

    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False

    rcParams["axes.linewidth"] = 2

    img_width = 10
    img_height = 9

    print(dataset_name)
    print('------------------------------------------------------------')
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    data_len = len(age_real)
    xlim = [min(age_real)-1, max(age_real)+1]
    X_input = np.linspace(min(age_real), max(age_real), 20)
    X_input_reshape = np.reshape(X_input, (20, 1))
    # real age
    #if method == 'linear':

    X = np.array(age_real)
    reg = LinearRegression().fit(np.reshape(X, (data_len, 1)),
                                 regression_index)
    a = reg.coef_
    b = reg.intercept_
    R2_score = reg.score(np.reshape(X, (data_len, 1)), regression_index)
    R2_age = R2_score
    regression_index_real = reg.predict(np.reshape(X, (data_len, 1)))
    plot_Y = reg.predict(X_input_reshape)
    MSE_age = mean_squared_error(regression_index, regression_index_real)
    print('Real age, a = {}, b = {}, R2 = {}'.format(a, b, R2_score))
    #plt.plot(X, regression_index_real, '-', color=(50 / 255, 50 / 255, 50 / 255), label='Age',
    #         linewidth=2)
    plt.plot(X_input, plot_Y, '-', color=(50 / 255, 50 / 255, 50 / 255), label='Age',
             linewidth=2)

    X = np.array(age_predict)
    reg = LinearRegression().fit(np.reshape(X, (data_len, 1)),
                                 regression_index)
    a = reg.coef_
    b = reg.intercept_
    R2_score = reg.score(np.reshape(X, (data_len, 1)), regression_index)
    R2_age_predict = R2_score
    regression_index_predict_age = reg.predict(np.reshape(X, (data_len, 1)))
    plot_Y = reg.predict(X_input_reshape)
    MSE_age_predict = mean_squared_error(regression_index, regression_index_predict_age)
    print('Predict age, a = {}, b = {}, R2 = {}'.format(a, b, R2_score))
    #plt.plot(X, regression_index_predict_age, '-+', color=(181 / 255, 85 / 255, 250 / 255), label='Predicted Age',
    #         linewidth=2)
    plt.plot(X_input, plot_Y, '-+', color=(181 / 255, 85 / 255, 250 / 255), label='Predicted Age',
             linewidth=2)


    # apad
    X = np.array(age_real) + np.array(apad)
    reg = LinearRegression().fit(np.reshape(X, (data_len, 1)),
                                 regression_index)
    a = reg.coef_
    b = reg.intercept_
    R2_score = reg.score(np.reshape(X, (data_len, 1)), regression_index)
    R2_apad = R2_score
    regression_index_predict_apad = reg.predict(np.reshape(X, (data_len, 1)))
    plot_Y = reg.predict(X_input_reshape)
    MSE_apad = mean_squared_error(regression_index, regression_index_predict_apad)
    print('Age-level correction, a = {}, b = {}, R2 = {}'.format(a, b, R2_score))
    plt.scatter(age_real, regression_index, color=(26 / 255, 85 / 255, 153 / 255), alpha=0.8, s=30)
    #plt.plot(X, regression_index_predict_apad, '--*', color=(255 / 255, 66 / 255, 14 / 255), label='Age-level', linewidth=2)
    plt.plot(X_input, plot_Y, '--*', color=(255 / 255, 66 / 255, 14 / 255), label='Age-level',
             linewidth=2)


    # linear apad
    X = np.array(age_real) + np.array(linear_apad)
    reg = LinearRegression().fit(np.reshape(X, (data_len, 1)),
                                 regression_index)
    a = reg.coef_
    b = reg.intercept_
    R2_score = reg.score(np.reshape(X, (data_len, 1)), regression_index)
    R2_linear_apad = R2_score
    regression_index_predict_linear_apad = reg.predict(np.reshape(X, (data_len, 1)))
    plot_Y = reg.predict(X_input_reshape)
    MSE_linear_apad = mean_squared_error(regression_index, regression_index_predict_linear_apad)
    print('Age-level and linear correction, a = {}, b = {}, R2 = {}'.format(a, b, R2_score))
    #plt.plot(X, regression_index_predict_linear_apad, '-d', color=(137 / 255, 218 / 255, 89 / 255),
    #         label='Linear + Age-level', linewidth=2)
    plt.plot(X_input, plot_Y, '-d', color=(137 / 255, 218 / 255, 89 / 255),
             label='Linear + Age-level', linewidth=2)


    # quad apad
    X = np.array(age_real) + np.array(quad_apad)
    reg = LinearRegression().fit(np.reshape(X, (data_len, 1)),
                                 regression_index)
    a = reg.coef_
    b = reg.intercept_
    R2_score = reg.score(np.reshape(X, (data_len, 1)), regression_index)
    R2_quad_apad = R2_score
    regression_index_predict_quad_apad = reg.predict(np.reshape(X, (data_len, 1)))
    plot_Y = reg.predict(X_input_reshape)
    MSE_quad_apad = mean_squared_error(regression_index, regression_index_predict_quad_apad)
    print('Age-level and quadratic correction, a = {}, b = {}, R2 = {}'.format(a, b, R2_score))
    #plt.plot(X, regression_index_predict_quad_apad, '-.', color=(255 / 255, 187 / 255, 125 / 255),
    #         label='Quadratic + Age-level', linewidth=2)
    plt.plot(X_input, plot_Y, '-.', color=(255 / 255, 187 / 255, 125 / 255),
             label='Quadratic + Age-level', linewidth=2)

    # linear pad
    X = np.array(age_real) + np.array(age_pad_linear)
    reg = LinearRegression().fit(np.reshape(X, (data_len, 1)),
                                 regression_index)
    a = reg.coef_
    b = reg.intercept_
    R2_score = reg.score(np.reshape(X, (data_len, 1)), regression_index)
    R2_linear_pad = R2_score
    regression_index_predict_linear_pad = reg.predict(np.reshape(X, (data_len, 1)))
    plot_Y = reg.predict(X_input_reshape)
    MSE_linear_pad = mean_squared_error(regression_index, regression_index_predict_linear_pad)
    print('Linear correction, a = {}, b = {}, R2 = {}'.format(a, b, R2_score))
    plt.plot(X_input, plot_Y, '-s', color=(76 / 255, 181 / 255, 245 / 255),
             label='Linear', linewidth=2)

    # quad pad
    X = np.array(age_real) + np.array(age_pad_quad)
    reg = LinearRegression().fit(np.reshape(X, (data_len, 1)),
                                 regression_index)
    a = reg.coef_
    b = reg.intercept_
    R2_score = reg.score(np.reshape(X, (data_len, 1)), regression_index)
    R2_quad_pad = R2_score
    regression_index_predict_quad_pad = reg.predict(np.reshape(X, (data_len, 1)))
    plot_Y = reg.predict(X_input_reshape)
    MSE_quad_pad = mean_squared_error(regression_index, regression_index_predict_quad_pad)
    print('Quadratic correction, a = {}, b = {}, R2 = {}'.format(a, b, R2_score))
    #plt.plot(X, regression_index_predict_quad_pad, '--', color=(189 / 255, 124 / 255, 119 / 255),
    #         label='Quadratic', linewidth=2)
    plt.plot(X_input, plot_Y, '--', color=(189 / 255, 124 / 255, 119 / 255),
             label='Quadratic', linewidth=2)


    plt.ticklabel_format(style='sci', scilimits=(0, 0))
    #plt.subplots_adjust(bottom=0.15, left=0.20)  #
    # plt.plot(age_range_list, age_linear_vec, color= (76/255, 0, 9/255), linewidth = 3)
    # plt.title("PAD and chronological age")
    if dataset_name == 'ABIDE' and regression_index_name in ['SRS_awareness', 'SRS_cognition', 'SRS_communication', 'SRS_motivation', 'SRS_manierisms']:
        print(set(list(['SRS_awareness', 'SRS_cognition', 'SRS_communication', 'SRS_motivation', 'SRS_manierisms'])))
        plt.xlabel("Chronological or Corrected Age")
    if dataset_name == 'OASIS' and regression_index_name in ['SubCortGrayVol', 'SupraTentorialVol', 'lhCorticalWhiteMatterVol', 'rhCorticalWhiteMatterVol', 'CorticalWhiteMatterVol']:
        plt.xlabel("Chronological or Corrected Age")
    plt.ylabel(regression_index_display_name)
    if dataset_name == 'ABIDE' and regression_index_name == 'FIQ':
        plt.legend(loc='best')
    if dataset_name == 'OASIS' and regression_index_name == 'IntraCranialVol':
        plt.legend(loc='best')
    # plt.tight_layout()
    plt.xlim(xlim)
    plt.margins(x=0.2, y=0.1)
    # plt.ylim((-20, 20))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + '{}_PAD_endpoints.svg'.format(regression_index_name), format='svg', transparent=True)

    return [R2_age, R2_age_predict, R2_apad, R2_linear_apad, R2_quad_apad, R2_linear_pad, R2_quad_pad], [MSE_age, MSE_age_predict, MSE_apad, MSE_linear_apad, MSE_quad_apad, MSE_linear_pad, MSE_quad_pad]


def linear_regression_endpoints_v2(dataset_name, age_real, age_predict, apad, linear_apad, quad_apad,
                                   age_pad, age_pad_linear, age_pad_quad, regression_index,
                                   regression_index_name, regression_index_display_name, save_path, method='linear'):
    import seaborn as sns
    from statannot import add_stat_annotation
    sns.set(style='whitegrid')
    TINY_SIZE = 20  # 39
    SMALL_SIZE = 22  # 42
    MEDIUM_SIZE = 26  # 46
    BIGGER_SIZE = 26  # 46

    plt.rc('font', size=18)  # 35 controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    # plt.rc('legend',**{'fontsize':16})
    rcParams["axes.linewidth"] = 1
    rcParams["legend.frameon"] = True

    img_width = 11
    img_height = 7

    print(dataset_name)
    print('------------------------------------------------------------')
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    ax = plt.gca()
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

    data_len = len(age_real)
    '''
    xlim = [min(age_real) - 1, max(age_real) + 1]
    X_input = np.linspace(min(age_real), max(age_real), 20)
    X_input_reshape = np.reshape(X_input, (20, 1))
    # real age
    # if method == 'linear':
    
    X = np.array(age_real)
    reg = LinearRegression().fit(np.reshape(X, (data_len, 1)),
                                 regression_index)
    a = reg.coef_
    b = reg.intercept_
    R2_score = reg.score(np.reshape(X, (data_len, 1)), regression_index)
    R2_age = R2_score
    regression_index_real = reg.predict(np.reshape(X, (data_len, 1)))
    plot_Y = reg.predict(X_input_reshape)
    MSE_age = mean_squared_error(regression_index, regression_index_real)
    print('Real age, a = {}, b = {}, R2 = {}'.format(a, b, R2_score))
    # plt.plot(X, regression_index_real, '-', color=(50 / 255, 50 / 255, 50 / 255), label='Age',
    #         linewidth=2)
    plt.plot(X_input, plot_Y, '-', color=(50 / 255, 50 / 255, 50 / 255), label='Age',
             linewidth=2)
    '''
    df_output_dict = {}
    df_output_dict['Method'] = []
    df_output_dict['R2'] = []
    df_output_dict['R2_relative'] = []
    df_output_dict['R2_relative_PAD'] = []
    df_output_dict['R2_PAD'] = []
    df_output_dict['Coefficients_relative'] = []
    df_output_dict['MSE'] = []
    # PAD
    df_output_dict = evaluate_regression_PAD(age_real, age_pad, regression_index, df_output_dict, method='Uncorrected')
    # Age-level
    df_output_dict = evaluate_regression_PAD(age_real, apad, regression_index, df_output_dict, method='Age-level')
    # Linear+Age-level
    df_output_dict = evaluate_regression_PAD(age_real, linear_apad, regression_index, df_output_dict, method='Linear+AL')
    # Quadratic+Age-level
    df_output_dict = evaluate_regression_PAD(age_real, quad_apad, regression_index, df_output_dict, method='Quad+AL')
    # Linear
    df_output_dict = evaluate_regression_PAD(age_real, age_pad_linear, regression_index, df_output_dict, method='Linear')
    # Quadratic
    df_output_dict = evaluate_regression_PAD(age_real, age_pad_quad, regression_index, df_output_dict, method='Quadratic')

    df_output = pd.DataFrame.from_dict(df_output_dict)

    pal_dict = {'Age': (153 / 255, 153 / 255, 153 / 255), 'Uncorrected': (181 / 255, 85 / 255, 250 / 255),
                'Age-level': (255 / 255, 66 / 255, 14 / 255),
                'Linear+AL': (137 / 255, 218 / 255, 89 / 255), 'Quad+AL': (255 / 255, 184 / 255, 0 / 255),
                'Linear': (76 / 255, 181 / 255, 245 / 255), 'Quadratic': (189 / 255, 124 / 255, 119 / 255)}
    x = 'Method'
    order = ['Uncorrected', 'Age-level', 'Linear+AL', 'Quad+AL', 'Linear', 'Quadratic']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    y = 'R2'
    ax = sns.barplot(data=df_output, x=x, y=y, order=order, palette=pal_dict)
    # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
    plt.ylabel(r'$R^2$')
    plt.savefig(save_path + '{}_PAD_R2.svg'.format(regression_index_name), format='svg', transparent=True)

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    y = 'R2_relative'
    ax = sns.barplot(data=df_output, x=x, y=y, order=order, palette=pal_dict)
    # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
    plt.ylabel(r'Relative $R^2$')
    plt.savefig(save_path + '{}_PAD_R2_relative.svg'.format(regression_index_name), format='svg', transparent=True)

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    y = 'R2_PAD'
    ax = sns.barplot(data=df_output, x=x, y=y, order=order, palette=pal_dict)
    # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
    plt.ylabel(r'$R^2$ of PAD')
    plt.savefig(save_path + '{}_PAD_R2_PAD.svg'.format(regression_index_name), format='svg', transparent=True)

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    y = 'R2_relative_PAD'
    ax = sns.barplot(data=df_output, x=x, y=y, order=order, palette=pal_dict)
    # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
    plt.ylabel(r'Relative $R^2$ of PAD')
    plt.savefig(save_path + '{}_PAD_R2_relative_PAD.svg'.format(regression_index_name), format='svg', transparent=True)

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    y = 'Coefficients_relative'
    ax = sns.barplot(data=df_output, x=x, y=y, order=order, palette=pal_dict)
    # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
    plt.ylabel(r'Relative Regression Coefficient')
    plt.savefig(save_path + '{}_PAD_coefficient_relative.svg'.format(regression_index_name), format='svg', transparent=True)

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    y = 'MSE'
    ax = sns.barplot(data=df_output, x=x, y=y, order=order, palette=pal_dict)
    # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
    plt.ylabel(r'$MSE$')
    plt.savefig(save_path + '{}_PAD_MSE.svg'.format(regression_index_name), format='svg',
                transparent=True)


    #plt.ticklabel_format(style='sci', scilimits=(0, 0))
    # plt.subplots_adjust(bottom=0.15, left=0.20)  #
    # plt.plot(age_range_list, age_linear_vec, color= (76/255, 0, 9/255), linewidth = 3)
    # plt.title("PAD and chronological age")
    if dataset_name == 'ABIDE' and regression_index_name in ['SRS_awareness', 'SRS_cognition', 'SRS_communication',
                                                             'SRS_motivation', 'SRS_manierisms']:
        print(set(list(['SRS_awareness', 'SRS_cognition', 'SRS_communication', 'SRS_motivation', 'SRS_manierisms'])))
        plt.xlabel("Bias Correction Method")
    if dataset_name == 'OASIS' and regression_index_name in ['SubCortGrayVol', 'SupraTentorialVol',
                                                             'lhCorticalWhiteMatterVol', 'rhCorticalWhiteMatterVol',
                                                             'CorticalWhiteMatterVol']:
        plt.xlabel("Bias Correction Method")
    plt.ylabel(regression_index_display_name)
    '''
    if dataset_name == 'ABIDE' and regression_index_name == 'FIQ':
        plt.legend(loc='best')
    if dataset_name == 'OASIS' and regression_index_name == 'IntraCranialVol':
        plt.legend(loc='best')
    '''
    return df_output

import seaborn as sns
from statannot import add_stat_annotation

def linear_regression_endpoints_v3(dataset_name, age_real, age_predict, apad, linear_apad, quad_apad,
                                   age_pad, age_pad_linear, age_pad_quad, regression_index,
                                   regression_index_name, regression_index_display_name, regression_index_id, save_path, method='v1'):
    sns.set(style='white')
    TINY_SIZE = 32 # 39
    SMALL_SIZE = 34  # 42
    MEDIUM_SIZE = 42  # 46
    BIGGER_SIZE = 42  # 46

    plt.rc('font', size=30)  # 35 controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    # plt.rc('legend',**{'fontsize':16})
    rcParams["axes.linewidth"] = 1
    rcParams["legend.frameon"] = False
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False

    img_width = 12
    img_height = 7.5

    print(dataset_name)
    print('------------------------------------------------------------')
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    data_len = len(age_real)
    #print('age_real length', data_len)

    df_output_dict = {}
    df_output_dict['Method'] = []
    df_output_dict['R2'] = []
    df_output_dict['F_pvalue'] = []
    df_output_dict['T_pvalue'] = []

    # PAD
    df_output_dict = evaluate_regression_PAD_v3(age_real, age_pad, regression_index, df_output_dict, method='Uncorrected')
    # Age-level
    df_output_dict = evaluate_regression_PAD_v3(age_real, apad, regression_index, df_output_dict, method='Age-level')
    # Linear+Age-level
    df_output_dict = evaluate_regression_PAD_v3(age_real, linear_apad, regression_index, df_output_dict,
                                             method='Linear+Age-level')
    # Quadratic+Age-level
    df_output_dict = evaluate_regression_PAD_v3(age_real, quad_apad, regression_index, df_output_dict, method='Quadratic+Age-level')
    # Linear
    df_output_dict = evaluate_regression_PAD_v3(age_real, age_pad_linear, regression_index, df_output_dict,
                                             method='Linear')
    # Quadratic
    df_output_dict = evaluate_regression_PAD_v3(age_real, age_pad_quad, regression_index, df_output_dict,
                                             method='Quadratic')

    df_output = pd.DataFrame.from_dict(df_output_dict)

    pal_dict = {'Age': (153 / 255, 153 / 255, 153 / 255), 'Uncorrected': (181 / 255, 85 / 255, 250 / 255),
                'Age-level': (255 / 255, 66 / 255, 14 / 255),
                'Linear+Age-level': (137 / 255, 218 / 255, 89 / 255), 'Quadratic+Age-level': (255 / 255, 184 / 255, 0 / 255),
                'Linear': (76 / 255, 181 / 255, 245 / 255), 'Quadratic': (189 / 255, 124 / 255, 119 / 255)}

    xtick_rotate_angle = 20
    bottom_adjust = 0.3
    left_adjust = 0.2
    ha_adjust = 'right'

    x = 'Method'
    order = ['Uncorrected', 'Age-level', 'Linear+Age-level', 'Quadratic+Age-level', 'Linear', 'Quadratic']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    y = 'R2'
    ax = sns.barplot(data=df_output, x=x, y=y, order=order, palette=pal_dict, edgecolor = 'dimgray')

    ##------set width-------------
    # Set these based on your column counts
    columncounts = [70] * 6

    # Maximum bar width is 1. Normalise counts to be in the interval 0-1. Need to supply a maximum possible count here as maxwidth
    def normaliseCounts(widths, maxwidth):
        widths = np.array(widths) / float(maxwidth)
        return widths

    widthbars = normaliseCounts(columncounts, 100)

    # Loop over the bars, and adjust the width (and position, to keep the bar centred)
    for bar, newwidth in zip(ax.patches, widthbars):
        bar_x = bar.get_x()
        width = bar.get_width()
        centre = bar_x + width / 2.

        bar.set_x(centre - newwidth / 2.)
        bar.set_width(newwidth)

    y_max = df_output[y].max() * 15/14
    y_min = df_output[y].min() * 9/10
    plt.ylim([y_min, y_max])
    plt.subplots_adjust(bottom=bottom_adjust, left=left_adjust)
    # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
    plt.xlabel('')
    if dataset_name == 'UK_Biobank' and regression_index_name in ['4080-0.1']: #, '23105-0.0'
        plt.ylabel(r'$R^2$')
    else:
        plt.ylabel('')
    if method == 'v1':
        plt.title('{}'.format(regression_index_display_name))
    for item in ax.get_xticklabels():
        item.set_rotation(xtick_rotate_angle)
    plt.setp(ax.xaxis.get_majorticklabels(), ha=ha_adjust, rotation_mode="anchor")

    plt.savefig(save_path + '{}_PAD_R2.svg'.format(regression_index_name), format='svg', transparent=True)


    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    y = 'F_pvalue'
    ax = sns.barplot(data=df_output, x=x, y=y, order=order, palette=pal_dict, edgecolor = 'dimgray')
    ##------set width-------------
    # Set these based on your column counts
    columncounts = [70] * 6

    # Maximum bar width is 1. Normalise counts to be in the interval 0-1. Need to supply a maximum possible count here as maxwidth
    def normaliseCounts(widths, maxwidth):
        widths = np.array(widths) / float(maxwidth)
        return widths

    widthbars = normaliseCounts(columncounts, 100)

    # Loop over the bars, and adjust the width (and position, to keep the bar centred)
    for bar, newwidth in zip(ax.patches, widthbars):
        bar_x = bar.get_x()
        width = bar.get_width()
        centre = bar_x + width / 2.

        bar.set_x(centre - newwidth / 2.)
        bar.set_width(newwidth)
    # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
    y_max = df_output[y].max() * 15 / 14
    y_min = df_output[y].min() * 9 / 10
    plt.ylim([y_min, y_max])
    plt.subplots_adjust(bottom=bottom_adjust, left=left_adjust)
    plt.xlabel('')
    if method == 'v1':
        plt.title('{}'.format(regression_index_display_name))
    if dataset_name == 'UK_Biobank' and regression_index_name in ['4080-0.1']: #, '23105-0.0'
        plt.ylabel(r'-log(p) of F-test')
    else:
        plt.ylabel('')
    #plt.title('{}'.format(regression_index_display_name))
    for item in ax.get_xticklabels():
        item.set_rotation(xtick_rotate_angle)
    plt.setp(ax.xaxis.get_majorticklabels(), ha=ha_adjust, rotation_mode="anchor")

    plt.savefig(save_path + '{}_PAD_F_pvalue.svg'.format(regression_index_name), format='svg',
                transparent=True)

    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    y = 'T_pvalue'
    ax = sns.barplot(data=df_output, x=x, y=y, order=order, palette=pal_dict, edgecolor = 'dimgray')

    ##------set width-------------
    # Set these based on your column counts
    columncounts = [70] * 6

    # Maximum bar width is 1. Normalise counts to be in the interval 0-1. Need to supply a maximum possible count here as maxwidth
    def normaliseCounts(widths, maxwidth):
        widths = np.array(widths) / float(maxwidth)
        return widths

    widthbars = normaliseCounts(columncounts, 100)

    # Loop over the bars, and adjust the width (and position, to keep the bar centred)
    for bar, newwidth in zip(ax.patches, widthbars):
        bar_x = bar.get_x()
        width = bar.get_width()
        centre = bar_x + width / 2.

        bar.set_x(centre - newwidth / 2.)
        bar.set_width(newwidth)

    y_max = df_output[y].max() * 15 / 14
    y_min = df_output[y].min() * 9 / 10
    plt.ylim([y_min, y_max])
    plt.subplots_adjust(bottom=bottom_adjust, left=left_adjust)
    # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
    plt.xlabel('')
    if dataset_name == 'UK_Biobank' and regression_index_name in ['4080-0.1']: #, '23105-0.0'
        plt.ylabel(r'-log(p) of t-test')
    else:
        plt.ylabel('')
    for item in ax.get_xticklabels():
        item.set_rotation(xtick_rotate_angle)
    plt.setp(ax.xaxis.get_majorticklabels(), ha=ha_adjust, rotation_mode="anchor")

    plt.title('{}'.format(regression_index_display_name))
    plt.savefig(save_path + '{}_PAD_T_pvalue.svg'.format(regression_index_name), format='svg',
                transparent=True)

    # plt.ticklabel_format(style='sci', scilimits=(0, 0))
    # plt.subplots_adjust(bottom=0.15, left=0.20)  #
    # plt.plot(age_range_list, age_linear_vec, color= (76/255, 0, 9/255), linewidth = 3)
    # plt.title("PAD and chronological age")
    if dataset_name == 'ABIDE' and regression_index_name in ['SRS_awareness', 'SRS_cognition', 'SRS_communication',
                                                             'SRS_motivation', 'SRS_manierisms']:
        print(set(list(['SRS_awareness', 'SRS_cognition', 'SRS_communication', 'SRS_motivation', 'SRS_manierisms'])))
        plt.xlabel("Bias Correction Method")
    if dataset_name == 'OASIS' and regression_index_name in ['SubCortGrayVol', 'SupraTentorialVol',
                                                             'lhCorticalWhiteMatterVol', 'rhCorticalWhiteMatterVol',
                                                             'CorticalWhiteMatterVol']:
        plt.xlabel("Bias Correction Method")





    '''
    if dataset_name == 'ABIDE' and regression_index_name == 'FIQ':
        plt.legend(loc='best')
    if dataset_name == 'OASIS' and regression_index_name == 'IntraCranialVol':
        plt.legend(loc='best')
    '''
    return df_output


def regression_endpoints_v1(regression_index_name, regression_index_display_name, df_indexs, save_path,
                         dataset_name, age_real, predict_age,  apad, linear_apad, quad_apad, age_pad, age_pad_linear, age_pad_quad):
    #regression_index_name = 'IntraCranialVol'
    #regression_index_display_name = 'Intra-cranial Volume'
    regression_index = df_indexs[regression_index_name]
    # print(len(regression_index))
    save_path = save_path + '{}/'.format(regression_index_name)

    R2_list, MSE_list = linear_regression_endpoints_v1(dataset_name, age_real, predict_age,  apad, linear_apad, quad_apad, age_pad, age_pad_linear, age_pad_quad, regression_index,
                                regression_index_name, regression_index_display_name, save_path)
    return R2_list, MSE_list

def regression_endpoints_v2(regression_index_name, regression_index_display_name, df_indexs, save_path,
                         dataset_name, age_real, predict_age,  apad, linear_apad, quad_apad, age_pad, age_pad_linear, age_pad_quad):
    #regression_index_name = 'IntraCranialVol'
    #regression_index_display_name = 'Intra-cranial Volume'
    regression_index = df_indexs[regression_index_name]
    # print(len(regression_index))
    save_path = save_path + '{}/'.format(regression_index_name)

    df_output = linear_regression_endpoints_v2(dataset_name, age_real, predict_age,  apad, linear_apad, quad_apad, age_pad, age_pad_linear, age_pad_quad, regression_index,
                                regression_index_name, regression_index_display_name, save_path)
    return df_output


def regression_endpoints_v3(regression_index_name, regression_index_display_name, regression_index_id, df_indexs, save_path,
                         dataset_name, age_real, predict_age,  apad, linear_apad, quad_apad, age_pad, age_pad_linear, age_pad_quad, method):
    #regression_index_name = 'IntraCranialVol'
    #regression_index_display_name = 'Intra-cranial Volume'
    regression_index = df_indexs[regression_index_name]
    # print(len(regression_index))
    save_path = save_path + '{}_{}_{}/'.format(regression_index_id, regression_index_name, regression_index_display_name)

    df_output = linear_regression_endpoints_v3(dataset_name, age_real, predict_age,  apad, linear_apad, quad_apad, age_pad, age_pad_linear, age_pad_quad, regression_index,
                                regression_index_name, regression_index_display_name, regression_index_id, save_path, method)
    return df_output


from scipy import stats
def t_test_PAD(left_list, right_list):
    '''
    return stats.ttest_ind_from_stats(
        np.mean(left_list),
        np.std(left_list),
        len(left_list),
        np.mean(right_list),
        np.std(right_list),
        len(right_list),
        equal_var=False,
    )
    '''
    return stats.ttest_ind(left_list, right_list)


def evaluate_regression_PAD(age_real, age_pad, regression_index, df_output_dict, method):
    data_len  = len(age_real)
    X_age_real_np = np.reshape(np.array(age_real), (data_len, 1))
    X_age_pad_np = np.reshape(np.array(age_pad), (data_len, 1))
    X = np.concatenate((X_age_real_np, X_age_pad_np), axis=1)
    reg = LinearRegression().fit(np.reshape(X, (data_len, 2)),
                                 regression_index)
    a = reg.coef_
    b = reg.intercept_
    regression_index_predict_age = reg.predict(X)
    R2_score = r2_score(regression_index, regression_index_predict_age)
    R2_score_relative = abs(r2_score(regression_index, a[1] * np.array(age_pad) + b)) / abs(r2_score(regression_index,
                                                                                            regression_index_predict_age))
    Coefficents_relative = abs(a[1]) / (abs(a[0]) + abs(a[1]))
    MSE_score = mean_squared_error(regression_index, regression_index_predict_age)

    reg_real = LinearRegression().fit(X_age_real_np, regression_index)
    regression_index_predict_real = reg_real.predict(X_age_real_np)
    R2_score_real = r2_score(regression_index, regression_index_predict_real)
    reg_pad = LinearRegression().fit(X_age_pad_np, regression_index)
    regression_index_predict_pad = reg_pad.predict(X_age_pad_np)
    R2_score_pad = r2_score(regression_index, regression_index_predict_pad)
    R2_score_relative_pad = abs(R2_score) / abs(R2_score_real)

    df_output_dict['Method'].append(method)
    df_output_dict['R2'].append(R2_score)
    df_output_dict['R2_relative'].append(R2_score_relative)
    df_output_dict['R2_relative_PAD'].append(R2_score_relative_pad)
    df_output_dict['R2_PAD'].append(R2_score_pad)
    df_output_dict['Coefficients_relative'].append(Coefficents_relative)
    df_output_dict['MSE'].append(MSE_score)

    return df_output_dict


def evaluate_regression_PAD_v3(age_real, age_pad, regression_index, df_output_dict, method):
    data_len = len(age_real)

    # Data dataframe
    data_dict = {}
    data_dict['Real age'] = list(age_real)
    #print('age_real', len(list(age_real)))
    data_dict['PAD'] = list(age_pad)
    #print('age_pad', len(list(age_pad)))
    data_dict['Y'] = list(regression_index)
    #print('regression_index', len(list(regression_index)))
    data_df = pd.DataFrame.from_dict(data_dict)

    Y = data_df['Y']
    X = data_df[['Real age', 'PAD']]
    X = sm.add_constant(X)
    model = sm.GLS(Y, X)
    results = model.fit()

    df_output_dict['Method'].append(method)
    df_output_dict['R2'].append(results.rsquared)
    df_output_dict['F_pvalue'].append(-np.log10(results.f_pvalue))
    df_output_dict['T_pvalue'].append(-np.log10(results.pvalues['PAD']))

    return df_output_dict