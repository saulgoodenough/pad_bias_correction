import sys

import envaluate_resnet34
import envaluate_resnet34_abide
import envaluate_resnet34_oasis

import envaluate_mse_resnet34
import envaluate_crossentropy_resnet34

import evaluate_msd3d
import evaluate_sfcn
import envaluate_lasso
import envaluate_svr
import envaluate_xgboost

sys.modules['envaluate_resnet34'].__dict__.clear()
sys.modules['envaluate_resnet34_abide'].__dict__.clear()
sys.modules['envaluate_resnet34_oasis'].__dict__.clear()

sys.modules['envaluate_mse_resnet34'].__dict__.clear()
sys.modules['envaluate_crossentropy_resnet34'].__dict__.clear()

sys.modules['evaluate_msd3d'].__dict__.clear()
sys.modules['evaluate_sfcn'].__dict__.clear()
sys.modules['envaluate_lasso'].__dict__.clear()
sys.modules['envaluate_svr'].__dict__.clear()
sys.modules['envaluate_xgboost'].__dict__.clear()


