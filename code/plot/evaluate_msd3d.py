from code.utils.plot_utils import age_statistics as age_statistics
from code.utils.plot_utils import compute_error as compute_error
from code.utils.plot_utils import compute_error_all as compute_error_all
from code.utils.plot_utils import compute_class_accuracy as compute_class_accuracy



# msd range
whole_path = '../../msd3d_predict/range/whole/2021-07-13_age_predict.csv'
#left_path = '../../sfcn_predict/left/2021-06-26_age_predict.csv'
#right_path = '../../sfcn_predict/right/2021-06-26_age_predict.csv'

age_statistics(whole_path, 'msd3d range 38-86')
compute_error_all(whole_path, 'msd3d range 38-86')
compute_error(whole_path, 'msd3d range 38-86')
compute_class_accuracy(whole_path, 'msd3d range 38-86')

# msd range sampler
whole_path = '../../msd3d_predict/range_sampler/whole/2021-07-13_age_predict.csv'
#left_path = '../../sfcn_predict/left/2021-06-26_age_predict.csv'
#right_path = '../../sfcn_predict/right/2021-06-26_age_predict.csv'
age_statistics(whole_path, 'sampled msd3d range 38-86')
compute_error_all(whole_path, 'sampled msd3d range 38-86')
compute_error(whole_path, 'sampled msd3d range 38-86')
compute_class_accuracy(whole_path, 'sampled msd3d range 38-86')

