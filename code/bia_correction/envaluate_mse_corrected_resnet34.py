from code.utils.plot_utils import plot_sequence_general as plot_sequence_general

from code.utils.bias_correction import correct_age as correct_age


## Correction: None
method = None

# resnet3d range
whole_path = '../../resnet3d34_predict/range/mse_adam_correction/2021-11-21_age_predict.csv'
plot_title = 'resnet3d34 range 38-86 with MSE loss'
save_path = '../../resnet3d34_predict/range/mse_adam_correction/2021-11-21_not_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence_general(whole_path_corrected, plot_title, save_path, ifcorrection = True, dataset_name='ukbiobank')


'''
# resnet3d range sampler
whole_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_age_predict.csv'
plot_title = 'resample resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_not_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)
'''



method = 'linear'

## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/mse_adam_correction/2021-11-21_age_predict.csv'
plot_title = 'resnet3d34 range 38-86 with MSE loss'
save_path = '../../resnet3d34_predict/range/mse_adam_correction/2021-11-21_linear_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence_general(whole_path_corrected, plot_title, save_path, ifcorrection = True, dataset_name='ukbiobank')

'''
# resnet3d range sampler
whole_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_age_predict.csv'
plot_title = 'resample resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_linear_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)
'''

method = 'square'
## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/mse_adam_correction/2021-11-21_age_predict.csv'
plot_title = 'resnet3d34 range 38-86 with MSE loss'
save_path = '../../resnet3d34_predict/range/mse_adam_correction/2021-11-21_square_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence_general(whole_path_corrected, plot_title, save_path, ifcorrection = True, dataset_name='ukbiobank')

'''
# resnet3d range sampler
whole_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_age_predict.csv'
plot_title = 'resample resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_square_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)
'''

method = 'threeOrder'
## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/mse_adam_correction/2021-11-21_age_predict.csv'
plot_title = 'resnet3d34 range 38-86 with MSE loss'
save_path = '../../resnet3d34_predict/range/mse_adam_correction/2021-11-21_threeOrder_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence_general(whole_path_corrected, plot_title, save_path, ifcorrection = True, dataset_name='ukbiobank')

'''
# resnet3d range sampler
whole_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_age_predict.csv'
plot_title = 'resample resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_threeOrder_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)
'''

method = 'fourOrder'
## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/mse_adam_correction/2021-11-21_age_predict.csv'
plot_title = 'resnet3d34 range 38-86 with MSE loss'
save_path = '../../resnet3d34_predict/range/mse_adam_correction/2021-11-21_fourOrder_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence_general(whole_path_corrected, plot_title, save_path, ifcorrection = True, dataset_name='ukbiobank')

'''
# resnet3d range sampler
whole_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_age_predict.csv'
plot_title = 'resample resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_fourOrder_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)
'''

'''
method = 'polynomial'
## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/mse_adam/2021-11-07_age_predict.csv'
plot_title = 'resnet3d34 range 38-86 with MSE loss'
save_path = '../../resnet3d34_predict/range/mse_adam/2021-11-07_polynomial_corrected/'
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
whole_path = '../../resnet3d34_predict/range/mse_adam/2021-11-07_age_predict.csv'
plot_title = 'resnet3d34 range 38-86 with MSE loss'
save_path = '../../resnet3d34_predict/range/mse_adam/2021-11-07_svr_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)


# resnet3d range sampler
whole_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_age_predict.csv'
plot_title = 'resample resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range_sampler_add1-2/whole/2021-10-02_svr_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)
'''

