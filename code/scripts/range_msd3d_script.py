import sys
sys.path.append('../')
from config import leftright_msd3d_range
from trainmodel import trainmodel
from predict import predict

cfg = leftright_msd3d_range._C
trained_model_dict =  trainmodel(cfg)
predict(cfg, trained_model_dict)
