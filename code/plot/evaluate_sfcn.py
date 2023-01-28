from code.utils.plot_utils import age_statistics_real as age_statistics_real
from code.utils.plot_utils import age_statistics as age_statistics
from code.utils.plot_utils import compute_error as compute_error
from code.utils.plot_utils import compute_error_all as compute_error_all
from code.utils.plot_utils import compute_class_accuracy as compute_class_accuracy


# sfcn
whole_path = '../../sfcn_predict/whole/2021-07-11_age_predict.csv'
#left_path = '../../sfcn_predict/left/2021-06-26_age_predict.csv'
#right_path = '../../sfcn_predict/right/2021-06-26_age_predict.csv'

age_statistics(whole_path, 'sfcn range 42-82')
age_statistics_real(whole_path)

#compute_error(whole_path, 'resnet3d range 42-82')
compute_error(whole_path, 'sfcn range 42-82')
compute_error_all(whole_path, 'sfcn range 42-82')
compute_class_accuracy(whole_path, 'sfcn range 42-82')
#plot_left_right(left_path, right_path, 'resnet3d range 42-82')

# sfcn range
whole_path = '../../sfcn_predict/range/whole/2021-07-05_age_predict.csv'
#left_path = '../../sfcn_predict/left/2021-06-26_age_predict.csv'
#right_path = '../../sfcn_predict/right/2021-06-26_age_predict.csv'

age_statistics(whole_path, 'sfcn range 38-86')
compute_error_all(whole_path, 'sfcn range 38-86')
compute_error(whole_path, 'sfcn range 38-86')
compute_class_accuracy(whole_path, 'sfcn range 38-86')
#plot_left_right(left_path, right_path, 'resnet3d range 42-82')


# sfcn sample
whole_path = '../../sfcn_predict/sampler/whole/2021-07-14_age_predict.csv'
#left_path = '../../sfcn_predict/left/2021-06-26_age_predict.csv'
#right_path = '../../sfcn_predict/right/2021-06-26_age_predict.csv'

age_statistics(whole_path, 'sampled sfcn range 42-82')
compute_error_all(whole_path, 'sampled sfcn range 42-82')
compute_error(whole_path, 'sampled sfcn range 42-82')
compute_class_accuracy(whole_path, 'sampled sfcn range 42-82')
#plot_left_right(left_path, right_path, 'resnet3d range 42-82')

