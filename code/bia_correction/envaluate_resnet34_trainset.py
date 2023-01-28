from code.utils.plot_utils import plot_sequence as plot_sequence

from code.utils.bias_correction import correct_age as correct_age


## Correction: None
method = None

# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/training_set/100_2021-11-17_trainingset_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/training_set/100_2021-11-17_not_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)





method = 'linear'

## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/training_set/100_2021-11-17_trainingset_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/training_set/100_2021-11-17_linear_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)






method = 'square'
## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/training_set/100_2021-11-17_trainingset_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/training_set/100_2021-11-17_square_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)



method = 'method2_linear'

## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/training_set/100_2021-11-17_trainingset_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/training_set/100_2021-11-17_method2_linear_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)






method = 'method2_square'
## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/training_set/100_2021-11-17_trainingset_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/training_set/100_2021-11-17_method2_square_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)

'''
method = 'threeOrder'
## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/training_set/100_2021-11-17_trainingset_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/training_set/100_2021-11-17_threeOrder_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)





method = 'fourOrder'
## Correction: None
# resnet3d range
whole_path = '../../resnet3d34_predict/range/lr1e-1/training_set/100_2021-11-17_trainingset_age_predict.csv'
plot_title = 'resnet3d34 range 38-86'
save_path = '../../resnet3d34_predict/range/lr1e-1/training_set/100_2021-11-17_fourOrder_corrected/'
whole_path_corrected = correct_age(whole_path, save_path, method = method)
plot_sequence(whole_path_corrected, plot_title, save_path)

'''



