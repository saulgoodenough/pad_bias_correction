from code.utils.plot_utils import plot_sequence as plot_sequence

from code.utils.plot_utils import bar_mean_all

from code.utils.bias_correction import correct_age as correct_age


## Correction: None
method = None

# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_not_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
corr_list, corr_list_zscore = plot_sequence(whole_path_corrected, plot_title, save_path, if_bar_xlabel = False, if_bar_ylabel = True, if_scatter_xlabel = False, if_scatter_ylabel = True)



method = 'linear'

## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_linear_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
corr_list_linear1, corr_list_linear1_zscore = plot_sequence(whole_path_corrected, plot_title, save_path, if_bar_xlabel = False, if_bar_ylabel = True, if_scatter_xlabel = False, if_scatter_ylabel = True)



method = 'square'
## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_square_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
corr_list_square1, corr_list_square1_zscore = plot_sequence(whole_path_corrected, plot_title, save_path, if_bar_xlabel = False, if_bar_ylabel = True, if_scatter_xlabel = True,if_scatter_ylabel = True)


bar_mean_all(corr_list, corr_list_linear1, corr_list_square1, corr_list_zscore, corr_list_linear1_zscore, corr_list_square1_zscore, save_path, if_bar_xlabel=False, if_bar_ylabel=True)



method = 'method2_linear'

## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_method2_linear_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
corr_list_linear2, corr_list_linear2_zscore  = plot_sequence(whole_path_corrected, plot_title, save_path, if_bar_xlabel = False, if_bar_ylabel = True, if_scatter_xlabel = False, if_scatter_ylabel = True)



method = 'method2_square'
## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_method2_square_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
corr_list_square2, corr_list_square2_zscore = plot_sequence(whole_path_corrected, plot_title, save_path, if_bar_xlabel = False, if_bar_ylabel = True, if_scatter_xlabel = True, if_scatter_ylabel = True)


bar_mean_all(corr_list, corr_list_linear2, corr_list_square2, corr_list_zscore, corr_list_linear2_zscore, corr_list_square2_zscore,  save_path, if_bar_xlabel=False, if_bar_ylabel=True)

'''

# resnet3d range sampler
whole_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_age_predict.csv'
plot_title = 'resample resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_square_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)


method = 'threeOrder'
## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_threeOrder_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)



# resnet3d range sampler
whole_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_age_predict.csv'
plot_title = 'resample resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_threeOrder_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)


method = 'fourOrder'
## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_fourOrder_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)



# resnet3d range sampler
whole_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_age_predict.csv'
plot_title = 'resample resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_fourOrder_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)



method = 'polynomial'
## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_polynomial_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)



# resnet3d range sampler
whole_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_age_predict.csv'
plot_title = 'resample resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_polynomial_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)


method = 'svr'
## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/2021-11-17_svr_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)


# resnet3d range sampler
whole_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_age_predict.csv'
plot_title = 'resample resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_svr_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)


'''