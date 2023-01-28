from code.utils.plot_utils import age_statistics_real as age_statistics_real
from code.utils.plot_utils import age_statistics as age_statistics
from code.utils.plot_utils import compute_error as compute_error
from code.utils.plot_utils import compute_error_all as compute_error_all
from code.utils.plot_utils import compute_class_accuracy as compute_class_accuracy


'''
# resnet3d
whole_path = '../../resnet3d_predict/whole/2021-07-10_age_predict.csv'
age_statistics(whole_path, 'resnet3d range 42-82')
compute_error(whole_path, 'resnet3d range 42-82')
compute_error_all(whole_path, 'resnet3d range 42-82')
compute_class_accuracy(whole_path, 'resnet3d range 42-82')

# resnet3d sampler
whole_path = '../../resnet3d_predict/sampler/whole/2021-07-12_age_predict.csv'
age_statistics(whole_path, 'sample resnet3d range 42-82')
compute_error(whole_path, 'sample resnet3d range 42-82')
compute_error_all(whole_path, 'sample resnet3d range 42-82')
compute_class_accuracy(whole_path, 'sample resnet3d range 42-82')
'''

# resnet3d range
whole_path = '../../resnet3d34_predict/range/sgd/2021-09-17_age_predict.csv'
save_path = '../../resnet3d34_predict/range/sgd/'
age_statistics_real(whole_path)
age_statistics(whole_path, 'resnet3d34_range 38-86')
compute_error(whole_path, 'resnet3d34_range 38-86')
compute_error_all(whole_path, 'resnet3d34_range 38-86')
compute_class_accuracy(whole_path, 'resnet3d34_range 38-86')

# resnet3d range sampler
whole_path = '../../resnet3d34_predict/range_sampler/whole/2021-09-22_age_predict.csv'
age_statistics(whole_path, 'sample resnet3d34 range 38-86')
compute_error(whole_path, 'sample resnet3d34 range 38-86')
compute_error_all(whole_path, 'sample resnet3d34 range 38-86')
compute_class_accuracy(whole_path, 'sample resnet3d34 range 38-86')



