from code.utils.plot_utils import plot_sequence as plot_sequence

from code.utils.bias_correction import correct_age as correct_age
from code.utils.plot_utils import bar_mean_all


## Correction: None
method = None

# msd3d range
whole_path = '../../msd3d_predict/range/whole/2021-08-10_age_predict.csv'
plot_title = 'msd3d range 38-86'
save_path = '../../msd3d_predict/range/whole/2021-08-10_not_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
corr_list, corr_list_zscore = plot_sequence(whole_path_corrected, plot_title, save_path)


method = 'linear'

## Correction: None
# msd3d range
whole_path = '../../msd3d_predict/range/whole/2021-08-10_age_predict.csv'
plot_title = 'resample msd3d range 38-86'
save_path = '../../msd3d_predict/range/whole/2021-08-10_linear_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
corr_list_linear1, corr_list_linear1_zscore = plot_sequence(whole_path_corrected, plot_title, save_path)


method = 'square'

## Correction: None
# msd3d range
whole_path = '../../msd3d_predict/range/whole/2021-08-10_age_predict.csv'
plot_title = 'resample msd3d range 38-86'
save_path = '../../msd3d_predict/range/whole/2021-08-10_square_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
corr_list_square1, corr_list_square1_zscore = plot_sequence(whole_path_corrected, plot_title, save_path, if_bar_xlabel = False, if_bar_ylabel = False, if_scatter_xlabel = True, if_scatter_ylabel = False)

bar_mean_all(corr_list, corr_list_linear1, corr_list_square1, corr_list_zscore, corr_list_linear1_zscore, corr_list_square1_zscore, save_path, if_bar_xlabel=False, if_bar_ylabel=True)


method = 'method2_linear'

## Correction: None
# msd3d range
whole_path = '../../msd3d_predict/range/whole/2021-08-10_age_predict.csv'
plot_title = 'resample msd3d range 38-86'
save_path = '../../msd3d_predict/range/whole/2021-08-10_method2_linear_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
corr_list_linear2, corr_list_linear2_zscore = plot_sequence(whole_path_corrected, plot_title, save_path)


method = 'method2_square'

## Correction: None
# msd3d range
whole_path = '../../msd3d_predict/range/whole/2021-08-10_age_predict.csv'
plot_title = 'resample msd3d range 38-86'
save_path = '../../msd3d_predict/range/whole/2021-08-10_method2_square_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
corr_list_square2, corr_list_square2_zscore = plot_sequence(whole_path_corrected, plot_title, save_path, if_bar_xlabel = False, if_bar_ylabel = False, if_scatter_xlabel = True, if_scatter_ylabel = False)

bar_mean_all(corr_list, corr_list_linear2, corr_list_square2, corr_list_zscore, corr_list_linear2_zscore, corr_list_square2_zscore, save_path, if_bar_xlabel=False, if_bar_ylabel=True)
