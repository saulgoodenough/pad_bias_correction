from yacs.config import CfgNode as CN


# --------------------------------------------------------------
'''
_C.NET3D: Hyper paramters for general net

'''
# --------------------------------------------------------------
# Config of model
# --------------------------------------------------------------
_C = CN()

_C.NET_MODEL = CN()

_C.NET_MODEL.TYPE = 'MSD3D' # 'MSD3D', 'RESNET3D'
_C.NET_MODEL.CLASS_NUM = 48

_C.NET_MODEL.OUT_CHANNELS = 1

_C.MSD_MODEL = CN()
_C.MSD_MODEL.GROWTH_RATE = 1
_C.MSD_MODEL.KERNEL_SIZE = 3
_C.MSD_MODEL.DILATION_MOD = [3, 6, 3]
_C.MSD_MODEL.LAYERS_NUMBER = [3, 6, 3]

# --------------------------------------------------------------
# Config of INPUT
# --------------------------------------------------------------
_C.NET_INPUT = CN()
_C.NET_INPUT.IN_CHANNELS = 1
_C.NET_INPUT.IMAGE_SIZE = (162, 226, 209)
_C.NET_INPUT.CROP_SIZE = (160, 192, 160)

# --------------------------------------------------------------
# Config of INPUT
# --------------------------------------------------------------

# --------------------------------------------------------------
# Config of DATASETS
# --------------------------------------------------------------
_C.NET_DATASETS = CN()

# --------------------------------------------------------------
# Config of DATALOADER
# --------------------------------------------------------------
_C.NET_DATALOADER = CN()
_C.NET_DATALOADER.ENDLIST = ["T1.nii.gz"]
_C.NET_DATALOADER.IMAGE_DIR = '/home2/migrate_handan/Biobank_images/'#'../data/'
_C.NET_DATALOADER.TARGET_DICT = '../../data/age_dict.npy'

_C.NET_DATALOADER.TRAIN_IMAGELIST_PATH = '../../data/image_list_train.npy'
_C.NET_DATALOADER.TEST_IMAGELIST_PATH = '../../data/image_list_test.npy'
_C.NET_DATALOADER.TEST_PERCENT = 0.15

## age range setting
_C.NET_DATALOADER.BIN_RANGE = [38, 86]
_C.NET_DATALOADER.BIN_STEP =  1
_C.NET_DATALOADER.SIGMA = 1

_C.NET_DATALOADER.NUM_WORKERS = 32
_C.NET_DATALOADER.BATCH_SIZE = 4 # default 8

_C.NET_DATALOADER.TEST_BATCH_SIZE = 1

# set pin_memory to TRUE when GPU memory is rich
_C.NET_DATALOADER.PIN_MEMORY = False
#
_C.NET_DATALOADER.DROP_LAST = True

_C.NET_DATALOADER.TEST_BATCH_SIZE = 1 # default 8
_C.NET_DATALOADER.TEST_IMAGE_DIR = '../../data/TEST/'

_C.NET_DATALOADER.DATA_AUG = True

_C.NET_DATALOADER.DIRECTION = 'left_right'

# --------------------------------------------------------------
# Config of training
# --------------------------------------------------------------
_C.NET_TRAIN = CN()


_C.NET_TRAIN.TRAIN_TYPE = 'whole' # 'left_right'

_C.NET_TRAIN.BALANCE = False

_C.NET_TRAIN.DESCRIPTION = 'MSD-3d training'
_C.NET_TRAIN.USE_GPU = True#True
_C.NET_TRAIN.DEVICE_ID = '2,3'#'0,1,2'
_C.NET_TRAIN.DATA_DEVICE_ID = [0,1]#[0,1,2]
_C.NET_TRAIN.FINE_TUNE = False
_C.NET_TRAIN.PRETRAIN_MODEL_PATH = '../../msd3d_model/range/pytorch_model_22_2021'
_C.NET_TRAIN.WHOLE_PRETRAIN_MODEL_PATH = '../../msd3d_model/range/whole/pytorch_model_22_2021'
_C.NET_TRAIN.LEFT_PRETRAIN_MODEL_PATH = '../../msd3d_model/range/left/pytorch_model_22_2021'
_C.NET_TRAIN.RIGHT_PRETRAIN_MODEL_PATH = '../../msd3d_model/range/right/pytorch_model_22_2021'
_C.NET_TRAIN.EPOCHS = 40 #100
_C.NET_TRAIN.TEST_RATIO = 0.05

_C.NET_TRAIN.EARLY_STOPPING = False
_C.NET_TRAIN.PATIENCE = 10

_C.NET_TRAIN.VISDOM = False
# --------------------------------------------------------------
# Config of SOLVER
# --------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.SEED = 1921
_C.SOLVER.LOSS = 'KL' # 'MSE' 'W-Distance' ''

_C.SOLVER.NAME = 'SGD' # 'Adam'

_C.SOLVER.SGD_LR = 0.1
_C.SOLVER.SGD_WEIGHT_DECAY = 0.0001
_C.SOLVER.SGD_MOMENTUM = 0.9


_C.SOLVER.ADAM_LR = 0.001
_C.SOLVER.ADAM_BETAS = (0.9, 0.999)
_C.SOLVER.ADAM_EPS = 1e-08
_C.SOLVER.ADAM_WEIGHT_DECAY = 5*10**(-4)

if _C.SOLVER.NAME is 'SGD':
    _C.SOLVER.INIT_LR = _C.SOLVER.SGD_LR
elif _C.SOLVER.NAME is 'Adam':
    _C.SOLVER.INIT_LR = _C.SOLVER.ADAM_LR

_C.SOLVER.LR_DECAY = 0.1
_C.SOLVER.LR_PERIOD = 25

# --------------------------------------------------------------
# Config of TEST
# --------------------------------------------------------------
_C.NET_TEST = CN()

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.NET_TRAIN.NET_OUTPUT_DIR = "../../msd3d_model/range/"
_C.NET_TRAIN.WHOLE_NET_OUTPUT_DIR = "../../msd3d_model/range/whole/"
_C.NET_TRAIN.LEFT_NET_OUTPUT_DIR = "../../msd3d_model/range/left/"
_C.NET_TRAIN.RIGHT_NET_OUTPUT_DIR = "../../msd3d_model/range/right/"

_C.NET_PREDICT = CN()
_C.NET_PREDICT.WHOLE_MODEL = '../../msd3d_model/range/whole/pytorch_model_100_2021-06-28'
_C.NET_PREDICT.WHOLE_DIR = "../../msd3d_predict/range/whole/"
_C.NET_PREDICT.LEFT_MODEL = '../../msd3d_model/range/left/pytorch_model_100_2021-06-28'
_C.NET_PREDICT.LEFT_DIR = "../../msd3d_predict/range/left/"
_C.NET_PREDICT.RIGHT_MODEL = '../../msd3d_model/range/right/pytorch_model_100_2021-06-28'
_C.NET_PREDICT.RIGHT_DIR = "../../msd3d_predict/range/right/"

# ---------------------------------------------------------------------------- #
# Visualization
# ---------------------------------------------------------------------------- #
_C.VISUALIZER = CN()
_C.VISUALIZER.WHOLE_ENV = 'MSD_WHOLE'
_C.VISUALIZER.LEFT_ENV = 'MSD_LEFT'
_C.VISUALIZER.RIGHT_ENV = 'MSD_RIGHT'

