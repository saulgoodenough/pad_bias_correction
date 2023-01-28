import sys
sys.path.append('../')
from config_abide import resnet3d34_range_sgd
from trainmodel_general import trainmodel
from predict_general import predict

cfg = resnet3d34_range_sgd._C
predict_model_path = trainmodel(cfg)
predict(cfg, predict_model_path)
