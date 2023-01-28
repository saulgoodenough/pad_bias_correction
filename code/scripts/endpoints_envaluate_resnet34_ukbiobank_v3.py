import matplotlib.pyplot as plt
import pandas as pd

from code.utils.plot_utils import plot_sequence as plot_sequence
from code.utils.plot_utils import zscore_PAD_endpoint, regression_endpoints_v3, t_test_PAD

from code.utils.bias_correction import correct_age as correct_age
import numpy as np

import sys
from scipy import stats
from matplotlib import rcParams
import matplotlib
matplotlib.use('Agg')

if __name__ == "__main__":

    ## Correction: None
    method = None
    regression_method = 'linear'
    # resnet3d range
    whole_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_age_predict.csv'
    plot_title = 'resnet3d34 range 38-86'
    save_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_not_corrected/'
    whole_path_corrected = correct_age(whole_path, save_path, method = method)
    age_pad_df = pd.read_csv(whole_path_corrected)
    real_age_vec_zscore, age_diff_vec_zscore, real_age_index, predict_age_zscore, age_pad_vec = zscore_PAD_endpoint(age_pad_df)
    print('len of age_diff_vec_zscore = ', len(age_diff_vec_zscore))


    method = 'linear'
    save_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_linear_corrected/'
    whole_path_corrected = correct_age(whole_path, save_path, method = method)
    age_pad_linear_df = pd.read_csv(whole_path_corrected)
    real_age_vec_linear_zscore, age_diff_vec_linear_zscore, real_age_linear_index, predict_age_linear_zscore, age_pad_linear_vec = zscore_PAD_endpoint(age_pad_linear_df)
    print('len of age_diff_vec_linear_zscore = ', len(age_diff_vec_linear_zscore))


    method = 'square'
    save_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_square_corrected/'
    whole_path_corrected = correct_age(whole_path, save_path, method = method)
    age_pad_quad_df = pd.read_csv(whole_path_corrected)
    real_age_vec_quad_zscore, age_diff_vec_quad_zscore, real_age_quad_index, predict_age_quad_zscore, age_pad_quad_vec = zscore_PAD_endpoint(age_pad_quad_df)
    print('len of age_diff_vec_quad_zscore = ', len(age_diff_vec_quad_zscore))

    dataset_name = 'UK_Biobank'
    index_path = '../../../MRI_DATA/UK_Biobank/index_used_newid.csv'
    df_age_whole = pd.read_csv(index_path, sep=',', index_col=0, header=0)

    xtick_rotate_angle = 20
    bottom_adjust = 0.3
    left_adjust = 0.2
    ha_adjust = 'right'

    regression_index_name_list = ['4125-2.0', '4080-0.1', '21002-2.0', '4079-0.1', '23236-2.0', '21001-2.0', '49-2.0',
                           '20116-0.0', '30040-0.0',
                           '12702-2.0', '12687-2.1', '23235-2.0', '23120-0.0', '30050-0.0', '12673-2.0', '12682-2.0',
                           '102-2.0', '23226-2.0',
                           '1558-2.0', '20015-2.0', '30010-0.0', '23105-0.0', '137-2.0', '2443-2.0', '1249-2.0',
                           '30270-0.0', '23099-0.0', '20016-2.0',
                           '404-2.7', '20195-0.0', '20159-0.0', '3064-2.1', '22408-2.0', '1458-2.0', '20023-2.0',
                           '1970-0.0', '738-2.0', '20133-0.1', '709-0.0',
                           '1588-0.0', '1568-0.0']
    regression_index_display_name_list = ['Heel bone mineral density', 'Systolic blood pressure', 'Weight', 'Diastolic blood pressure',
                       'Total BMD (bone mineral density)',
                       'Body mass index (BMI)', 'Hip circumference', 'Still smoking', 'Mean corpuscular volume',
                       'Cardiac index during PWA',
                       'Mean srterial pressure during PWA', 'Total BMC (bone mineral content)', 'Arm fat mass (right)',
                       'Mean corpuscular haemoglobibin',
                       'Heart rate during PWA', 'Cardiac output during PWA', 'Pulse rate',
                       'Head BMD (bone mineral density)', 'Alcohol intake frequency',
                       'Sitting height', 'Red blood cell (erythrocyte) count', 'Basal metabolic rate',
                       'Number of treatments/medications taken', 'Diabetes diagnosed by doctor',
                       'Past tobacco smoking', 'Mean sphered cell volume',
                       'Duration to complete alphanumeric path (cognition)', 'Body fat percentage',
                       'Fluid intelligence score (cognition)', 'Duration to first press of snap-button (cognition)',
                       'Number of symbol digit matches attempted (cognition)',
                       'Number of symbol digit matches made correctly (cognition)', 'Peak expiratory flow (PEF)',
                       'ASA tissue volume',
                       'Cereal intake', 'Mean time to correctly identify matches (cognition)', 'Nervous feelings',
                       'Average total household income before tax',
                       'Time to complete round, pairs matching (cognition)', 'Number in household',
                       'Average weekly beer plus cider intake', 'Average weekly red wine intake']

    '''
    regression_index_display_name_list = ['Heel bone mineral density', 'Systolic blood pressure', 'Weight', 'Diastolic blood pressure',
                       'Total BMD (bone mineral density)',
                       'Body mass index (BMI)', 'Hip circumference', 'Still smoking', 'Mean corpuscular volume',
                       'Cardiac index during PWA',
                       'Mean srterial pressure during PWA', 'Total BMC (bone mineral content)', 'Arm fat mass (right)',
                       'Mean corpuscular haemoglobibin',
                       'Heart rate during PWA', 'Cardiac output during PWA', 'Pulse rate',
                       'Head BMD (bone mineral density)', 'Alcohol intake frequency',
                       'Sitting height', 'Red blood cell (erythrocyte) count', 'Basal metabolic rate',
                       'Number of treatments/medications taken', 'Diabetes diagnosed by doctor',
                       'Past tobacco smoking', 'Mean sphered cell volume',
                       'Duration to complete alphanumeric path (cognition)', 'Body fat percentage',
                       'Fluid intelligence score (cognition)', 'Duration to first press of snap-button (cognition)',
                       'Number of symbol digit matches attempted (cognition)',
                       'Number of symbol digit matches made correctly (cognition)', 'Peak expiratory flow (PEF)',
                       'Abdominal subcutaneous adipose tissue volume',
                       'Cereal intake', 'Mean time to correctly identify matches (cognition)', 'Nervous feelings',
                       'Average total household income before tax',
                       'Time to complete round, pairs matching (cognition)', 'Number in household',
                       'Average weekly beer plus cider intake', 'Average weekly red wine intake']
    '''

    save_path = '../../ukbiobank_dataset/endpoint_v3/method1_{}/'.format(regression_method)
    for i in range(len(regression_index_name_list)):

        regression_index_name = regression_index_name_list[i]
        regression_index_display_name = regression_index_display_name_list[i]
        regression_index_id = i

        print('Index name = {}'.format(regression_index_name))

        #df_age = df_age_whole[df_age_whole[regression_index_name] != -9999]
        df_age = df_age_whole[df_age_whole[regression_index_name].notnull()]

        # print('df_age.shape', df_age.shape)

        # print('real_age_index:', real_age_index)
        # print('df_age.index:', df_age.index)

        list_intersect = list(set(real_age_index) & set(df_age.index))

        print('len of list_intersect = ', list_intersect)

        index_df_predict = [list(real_age_index).index(x) for x in list_intersect]
        age_real = [real_age_vec_zscore[x] for x in index_df_predict]
        age_predict = [predict_age_zscore[x] for x in index_df_predict]
        apad = [age_diff_vec_zscore[x] for x in index_df_predict]
        linear_apad = [age_diff_vec_linear_zscore[x] for x in index_df_predict]
        quad_apad = [age_diff_vec_quad_zscore[x] for x in index_df_predict]

        age_pad = [age_pad_vec[x] for x in index_df_predict]
        age_pad_linear = [age_pad_linear_vec[x] for x in index_df_predict]
        age_pad_quad = [age_pad_quad_vec[x] for x in index_df_predict]

        real_age_index_sub = [real_age_index[x] for x in index_df_predict]
        df_indexs = df_age.loc[list_intersect]
        # print(df_indexs.index)

        print('Judge if the twi lists are equal:')
        k = 0
        for x, y in zip(real_age_index_sub, df_indexs.index):
            if x != y:
                k = 1
                print('The two lists are not aligned!')
        if k == 0:
            print('The two lists are aligned!')

        df_output = regression_endpoints_v3(regression_index_name,
                                            regression_index_display_name,
                                            regression_index_id,
                                            df_indexs,
                                            save_path,
                                            dataset_name,
                                            age_real,
                                            age_predict,
                                            apad,
                                            linear_apad,
                                            quad_apad,
                                            age_pad,
                                            age_pad_linear,
                                            age_pad_quad,
                                            method = 'v1')
        if i == 0:
            df_boxplot = df_output
        else:
            df_boxplot = pd.concat([df_boxplot, df_output])

    original_stdout = sys.stdout  # Save a reference to the original standard output

    # Save boxplot data
    import seaborn as sns
    from statannot import add_stat_annotation

    sns.set(style='white')
    TINY_SIZE = 32  # 39
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
    rcParams["legend.frameon"] = True

    img_width = 12
    img_height = 8
    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    ax = plt.gca()
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    order = ['Uncorrected', 'Age-level', 'Linear+Age-level', 'Quadratic+Age-level', 'Linear', 'Quadratic']
    pal_dict = {'Uncorrected': (181 / 255, 85 / 255, 250 / 255),
                'Age-level': (255 / 255, 66 / 255, 14 / 255),
                'Linear+Age-level': (137 / 255, 218 / 255, 89 / 255), 'Quadratic+Age-level': (255 / 255, 184 / 255, 0 / 255),
                'Linear': (76 / 255, 181 / 255, 245 / 255), 'Quadratic': (189 / 255, 124 / 255, 119 / 255)}

    x = 'Method'

    with open(save_path + 't_test_result.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(dataset_name)
        print('------------------------------------------------------------')

        plt.figure(figsize=(img_width, img_height))  # width:20, height:3
        y = 'R2'
        ax = sns.boxplot(data=df_boxplot, x=x, y=y, order=order, palette=pal_dict)
        '''
        add_stat_annotation(ax, data=df_boxplot, x=x, y=y, order=order,
                            box_pairs=[('None', 'Age-level'), ('Age-level', 'Linear+AL'),
                                       ('Age-level', 'Quad+AL'),
                                       ('Age-level', 'Linear'), ('Age-level', 'Quadratic')],
                            test='t-test_paired', text_format='star', loc='inside', verbose=2)
        '''
        # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
        plt.ylabel(r'$R^2$')
        for item in ax.get_xticklabels():
            item.set_rotation(xtick_rotate_angle)
        plt.setp(ax.xaxis.get_majorticklabels(), ha=ha_adjust, rotation_mode="anchor")
        plt.subplots_adjust(bottom=bottom_adjust, left=left_adjust)
        plt.savefig(save_path + '{}_PAD_R2.svg'.format(dataset_name), format='svg', transparent=True)

        plt.figure(figsize=(img_width, img_height))  # width:20, height:3
        y = 'F_pvalue'
        ax = sns.boxplot(data=df_boxplot, x=x, y=y, order=order, palette=pal_dict)
        '''
        add_stat_annotation(ax, data=df_boxplot, x=x, y=y, order=order,
                            box_pairs=[('None', 'Age-level'), ('Age-level', 'Linear+AL'),
                                       ('Age-level', 'Quad+AL'),
                                       ('Age-level', 'Linear'), ('Age-level', 'Quadratic')],
                            test='t-test_paired', text_format='star', loc='inside', verbose=2)
        '''
        # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
        plt.ylabel(r'-log(p) of F-test')
        for item in ax.get_xticklabels():
            item.set_rotation(xtick_rotate_angle)
        plt.setp(ax.xaxis.get_majorticklabels(), ha=ha_adjust, rotation_mode="anchor")
        plt.subplots_adjust(bottom=bottom_adjust, left=left_adjust)
        plt.savefig(save_path + '{}_PAD_F_pvalue.svg'.format(dataset_name), format='svg', transparent=True)

        plt.figure(figsize=(img_width, img_height))  # width:20, height:3
        y = 'T_pvalue'
        ax = sns.boxplot(data=df_boxplot, x=x, y=y, order=order, palette=pal_dict)
        '''
        add_stat_annotation(ax, data=df_boxplot, x=x, y=y, order=order,
                            box_pairs=[('None', 'Age-level'), ('Age-level', 'Linear+AL'),
                                       ('Age-level', 'Quad+AL'),
                                       ('Age-level', 'Linear'), ('Age-level', 'Quadratic')],
                            test='t-test_paired', text_format='star', loc='inside', verbose=2)
        '''
        # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
        plt.ylabel(r'-log(p) of t-test')
        for item in ax.get_xticklabels():
            item.set_rotation(xtick_rotate_angle)
        plt.setp(ax.xaxis.get_majorticklabels(), ha=ha_adjust, rotation_mode="anchor")

        plt.subplots_adjust(bottom=bottom_adjust, left=left_adjust)
        plt.savefig(save_path + '{}_PAD_T_pvalue.svg'.format(dataset_name), format='svg', transparent=True)

        sys.stdout = original_stdout

    #------------------Method 2------------------------------------------
    method = None
    bin_range = [42, 97]
    bar_lim = (-6, 4)
    # resnet3d range
    whole_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_age_predict.csv'
    plot_title = 'resnet3d34 oasis'
    save_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_not_corrected/'
    whole_path_corrected = correct_age(whole_path, save_path, method=method)
    age_pad_df = pd.read_csv(whole_path_corrected)
    real_age_vec_zscore, age_diff_vec_zscore, real_age_index, predict_age_zscore, age_pad = zscore_PAD_endpoint(age_pad_df)
    print('len of age_diff_vec_zscore = ', len(age_diff_vec_zscore))

    method = 'method2_linear'
    save_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_method2_linear_corrected/'
    whole_path_corrected = correct_age(whole_path, save_path, method=method)
    age_pad_linear_df = pd.read_csv(whole_path_corrected)
    real_age_vec_linear_zscore, age_diff_vec_linear_zscore, real_age_linear_index, predict_age_linear_zscore, age_pad_linear = zscore_PAD_endpoint(
        age_pad_linear_df)
    print('len of age_diff_vec_linear_zscore = ', len(age_diff_vec_linear_zscore))

    method = 'method2_square'
    save_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_method2_square_corrected/'
    whole_path_corrected = correct_age(whole_path, save_path, method=method)
    age_pad_quad_df = pd.read_csv(whole_path_corrected)
    real_age_vec_quad_zscore, age_diff_vec_quad_zscore, real_age_quad_index, predict_age_quad_zscore, age_pad_quad = zscore_PAD_endpoint(age_pad_quad_df)
    print('len of age_diff_vec_quad_zscore = ', len(age_diff_vec_quad_zscore))





    save_path = '../../ukbiobank_dataset/endpoint_v3/method2_{}/'.format(regression_method)
    for i in range(len(regression_index_name_list)):

        regression_index_name = regression_index_name_list[i]
        regression_index_display_name = regression_index_display_name_list[i]
        regression_index_id = i

        print('Index name = {}'.format(regression_index_name))

        #df_age = df_age_whole[df_age_whole[regression_index_name] != -9999]
        df_age = df_age_whole[df_age_whole[regression_index_name].notnull()]

        # print('df_age.shape', df_age.shape)

        # print('real_age_index:', real_age_index)
        # print('df_age.index:', df_age.index)

        list_intersect = list(set(real_age_index) & set(df_age.index))

        print('len of list_intersect = ', list_intersect)

        index_df_predict = [list(real_age_index).index(x) for x in list_intersect]
        age_real = [real_age_vec_zscore[x] for x in index_df_predict]
        age_predict = [predict_age_zscore[x] for x in index_df_predict]
        apad = [age_diff_vec_zscore[x] for x in index_df_predict]
        linear_apad = [age_diff_vec_linear_zscore[x] for x in index_df_predict]
        quad_apad = [age_diff_vec_quad_zscore[x] for x in index_df_predict]

        age_pad = [age_pad_vec[x] for x in index_df_predict]
        age_pad_linear = [age_pad_linear_vec[x] for x in index_df_predict]
        age_pad_quad = [age_pad_quad_vec[x] for x in index_df_predict]

        real_age_index_sub = [real_age_index[x] for x in index_df_predict]
        df_indexs = df_age.loc[list_intersect]
        # print(df_indexs.index)

        print('Judge if the twi lists are equal:')
        k = 0
        for x, y in zip(real_age_index_sub, df_indexs.index):
            if x != y:
                k = 1
                print('The two lists are not aligned!')
        if k == 0:
            print('The two lists are aligned!')

        df_output = regression_endpoints_v3(regression_index_name,
                                            regression_index_display_name,
                                            regression_index_id,
                                            df_indexs,
                                            save_path,
                                            dataset_name,
                                            age_real,
                                            age_predict,
                                            apad,
                                            linear_apad,
                                            quad_apad,
                                            age_pad,
                                            age_pad_linear,
                                            age_pad_quad,
                                            method = 'v2')
        if i == 0:
            df_boxplot = df_output
        else:
            df_boxplot = pd.concat([df_boxplot, df_output])

    original_stdout = sys.stdout  # Save a reference to the original standard output


    plt.figure(figsize=(img_width, img_height))  # width:20, height:3
    ax = plt.gca()
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

    x = 'Method'

    with open(save_path + 't_test_result.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(dataset_name)
        print('------------------------------------------------------------')

        plt.figure(figsize=(img_width, img_height))  # width:20, height:3
        y = 'R2'
        ax = sns.boxplot(data=df_boxplot, x=x, y=y, order=order, palette=pal_dict)
        '''
        add_stat_annotation(ax, data=df_boxplot, x=x, y=y, order=order,
                            box_pairs=[('None', 'Age-level'), ('Age-level', 'Linear+AL'),
                                       ('Age-level', 'Quad+AL'),
                                       ('Age-level', 'Linear'), ('Age-level', 'Quadratic')],
                            test='t-test_paired', text_format='star', loc='inside', verbose=2)
        '''
        # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
        plt.ylabel(r'$R^2$')
        for item in ax.get_xticklabels():
            item.set_rotation(xtick_rotate_angle)
        plt.setp(ax.xaxis.get_majorticklabels(), ha=ha_adjust, rotation_mode="anchor")

        plt.subplots_adjust(bottom=bottom_adjust, left=left_adjust)
        plt.savefig(save_path + '{}_PAD_R2.svg'.format(dataset_name), format='svg', transparent=True)

        plt.figure(figsize=(img_width, img_height))  # width:20, height:3
        y = 'F_pvalue'
        ax = sns.boxplot(data=df_boxplot, x=x, y=y, order=order, palette=pal_dict)
        '''
        add_stat_annotation(ax, data=df_boxplot, x=x, y=y, order=order,
                            box_pairs=[('None', 'Age-level'), ('Age-level', 'Linear+AL'),
                                       ('Age-level', 'Quad+AL'),
                                       ('Age-level', 'Linear'), ('Age-level', 'Quadratic')],
                            test='t-test_paired', text_format='star', loc='inside', verbose=2)
        '''
        # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
        plt.ylabel(r'-log(p) of F-test')
        for item in ax.get_xticklabels():
            item.set_rotation(xtick_rotate_angle)
        plt.setp(ax.xaxis.get_majorticklabels(), ha=ha_adjust, rotation_mode="anchor")

        plt.subplots_adjust(bottom=bottom_adjust, left=left_adjust)
        plt.savefig(save_path + '{}_PAD_F_pvalue.svg'.format(dataset_name), format='svg', transparent=True)

        plt.figure(figsize=(img_width, img_height))  # width:20, height:3
        y = 'T_pvalue'
        ax = sns.boxplot(data=df_boxplot, x=x, y=y, order=order, palette=pal_dict)
        '''
        add_stat_annotation(ax, data=df_boxplot, x=x, y=y, order=order,
                            box_pairs=[('None', 'Age-level'), ('Age-level', 'Linear+AL'),
                                       ('Age-level', 'Quad+AL'),
                                       ('Age-level', 'Linear'), ('Age-level', 'Quadratic')],
                            test='t-test_paired', text_format='star', loc='inside', verbose=2)
        '''
        # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
        plt.ylabel(r'-log(p) of t-test')
        for item in ax.get_xticklabels():
            item.set_rotation(xtick_rotate_angle)
        plt.setp(ax.xaxis.get_majorticklabels(), ha=ha_adjust, rotation_mode="anchor")

        plt.subplots_adjust(bottom=bottom_adjust, left=left_adjust)
        plt.savefig(save_path + '{}_PAD_T_pvalue.svg'.format(dataset_name), format='svg', transparent=True)

        sys.stdout = original_stdout

