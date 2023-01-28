from code.utils.plot_utils import plot_sequence as plot_sequence
from code.utils.plot_utils import bar_mean_all

from code.utils.bias_correction import correct_age as correct_age
import pandas as pd

## Correction: None
method = None
bin_range = [6, 65]
# resnet3d range
whole_path = '../../abide_dataset/predict/resnet3d34/2021-12-02_age_predict.csv'
#pd = pd.read_csv(whole_path)
#print(pd['real age'].values)

plot_title = 'resnet3d34 abide'
save_path = '../../abide_dataset/predict/resnet3d34/2021-12-02_not_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
corr_list, corr_list_zscore = plot_sequence(whole_path_corrected, plot_title, save_path, bin_range)


method = 'linear'

## Correction: None
# resnet3d range
whole_path = '../../abide_dataset/predict/resnet3d34/2021-12-02_age_predict.csv'
plot_title = 'resnet3d34 abide'
save_path = '../../abide_dataset/predict/resnet3d34/2021-12-02_linear_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
corr_list_linear1, corr_list_linear1_zscore = plot_sequence(whole_path_corrected, plot_title, save_path, bin_range)



method = 'square'
## Correction: None
# resnet3d range
whole_path = '../../abide_dataset/predict/resnet3d34/2021-12-02_age_predict.csv'
plot_title = 'resnet3d34 abide'
save_path = '../../abide_dataset/predict/resnet3d34/2021-12-02_square_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
corr_list_square1, corr_list_square1_zscore = plot_sequence(whole_path_corrected, plot_title, save_path, bin_range,  if_bar_xlabel = False, if_bar_ylabel = False, if_scatter_xlabel = True, if_scatter_ylabel = False)

bar_mean_all(corr_list, corr_list_linear1, corr_list_square1, corr_list_zscore, corr_list_linear1_zscore, corr_list_square1_zscore, save_path, if_bar_xlabel=True, if_bar_ylabel=True)


method = 'method2_linear'

## Correction: None
# resnet3d range
whole_path = '../../abide_dataset/predict/resnet3d34/2021-12-02_age_predict.csv'
plot_title = 'resnet3d34 abide'
save_path = '../../abide_dataset/predict/resnet3d34/2021-12-02_method2_linear_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
corr_list_linear2, corr_list_linear2_zscore = plot_sequence(whole_path_corrected, plot_title, save_path, bin_range)



method = 'method2_square'
## Correction: None
# resnet3d range
whole_path = '../../abide_dataset/predict/resnet3d34/2021-12-02_age_predict.csv'
plot_title = 'resnet3d34 abide'
save_path = '../../abide_dataset/predict/resnet3d34/2021-12-02_method2_square_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
corr_list_square2, corr_list_square2_zscore = plot_sequence(whole_path_corrected, plot_title, save_path, bin_range,  if_bar_xlabel = False, if_bar_ylabel = False, if_scatter_xlabel = True, if_scatter_ylabel = False)

bar_mean_all(corr_list, corr_list_linear2, corr_list_square2, corr_list_zscore, corr_list_linear2_zscore, corr_list_square2_zscore, save_path, if_bar_xlabel=True, if_bar_ylabel=True)

'''
method = 'threeOrder'
## Correction: None
# resnet3d range
whole_path = '../../abide_dataset/predict/resnet3d34/2021-12-02_age_predict.csv'
plot_title = 'resnet3d34 abide'
save_path = '../../abide_dataset/predict/resnet3d34/2021-12-02_threeOrder_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path, bin_range)
'''




