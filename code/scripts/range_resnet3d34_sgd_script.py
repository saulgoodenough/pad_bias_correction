import sys
sys.path.append('../')
from config import leftright_resnet3d34_range_sgd
from trainmodel import trainmodel
from predict import predict

cfg = leftright_resnet3d34_range_sgd._C
predict_model_path = trainmodel(cfg)
predict(cfg, predict_model_path)
