import sys
sys.path.append('../')
from config import leftright_resnet3d34_range_adam_mse
from trainmodel_mse import trainmodel_mse
from predict_mse import predict

cfg = leftright_resnet3d34_range_adam_mse._C
predict_model_path = trainmodel_mse(cfg)
#predict_model_path = '/home1/zhangbiao/multimodal/resnet3d34_model/range/mse_adam/pytorch_model_44_2021-11-04'
#trained_model_dict = {}
#trained_model_dict['whole'] = predict_model_path
predict(cfg, predict_model_path)
