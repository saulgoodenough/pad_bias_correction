import sys
sys.path.append('../')
from config import crossentropy_resnet3d34_range_sgd
from trainmodel import trainmodel
from predict import predict

cfg = crossentropy_resnet3d34_range_sgd._C
#predict_model_path = {}
#predict_model_path['whole'] = '/home1/zhangbiao/multimodal/resnet3d34_model/range/crossentropy/pytorch_model_49_2021-12-08'
predict_model_path = trainmodel(cfg)
predict(cfg, predict_model_path)
