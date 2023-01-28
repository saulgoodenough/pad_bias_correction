
import sys

sys.path.append('../')

from statistic_config import pca_config
from utils.data_utils import PCADataProcessor


cfg = pca_config._C

pca_processor = PCADataProcessor(cfg)


train_data, train_age_array, validation_data, validation_age_array, test_data, test_age_array = pca_processor.get_PCA_data()





