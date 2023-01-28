from code.utils.plot_utils import plot_sequence as plot_sequence

from code.utils.bias_correction import correct_age as correct_age


## Correction: None
method = None

# resnet3d range
whole_path = '../../resnet3d_predict/range/whole/2021-08-04_age_predict.csv'
plot_title = 'resnet3d_range 38-86'
save_path = '../../resnet3d_predict/range/whole/2021-08-04_not_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)


# resnet3d range sampler
whole_path = '../../resnet3d_predict/range_sampler/whole/2021-08-06_age_predict.csv'
plot_title = 'resample resnet3d_range 38-86'
save_path = '../../resnet3d_predict/range_sampler/whole/2021-08-06_not_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)


method = 'linear'

## Correction: None
# resnet3d range
whole_path = '../../resnet3d_predict/range/whole/2021-08-04_age_predict.csv'
plot_title = 'resnet3d_range 38-86'
save_path = '../../resnet3d_predict/range/whole/2021-08-04_linear_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)



# resnet3d range sampler
whole_path = '../../resnet3d_predict/range_sampler/whole/2021-08-06_age_predict.csv'
plot_title = 'resample resnet3d_range 38-86'
save_path = '../../resnet3d_predict/range_sampler/whole/2021-08-06_linear_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)

