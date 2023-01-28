import sys
sys.path.append('../')
from config import leftright_sfcn_range
from trainmodel import trainmodel
from predict import predict

cfg = leftright_sfcn_range._C
predict_model_path = trainmodel(cfg)
predict(cfg, predict_model_path)
