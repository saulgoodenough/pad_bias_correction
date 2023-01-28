from yacs.config import CfgNode as CN


# --------------------------------------------------------------
'''
_C.MSD3D: Hyper paramters for MSD net

'''
# --------------------------------------------------------------
# Config of model
# --------------------------------------------------------------
_C = CN()

_C.MSD_MODEL = CN()

_C.MSD_MODEL.TYPE = 'MSD3D_ADD'
_C.MSD_MODEL.CLASS_NUM = 40

_C.MSD_MODEL.OUT_CHANNELS = 1

_C.MSD_MODEL.GROWTH_RATE = 1
_C.MSD_MODEL.KERNEL_SIZE = 3
_C.MSD_MODEL.DILATION_MOD = [3, 3, 2, 2]

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
_C.MSD_DATASETS = CN()

# --------------------------------------------------------------
# Config of DATALOADER
# --------------------------------------------------------------
_C.NET_DATALOADER = CN()
_C.NET_DATALOADER.ENDLIST = ["T1.nii.gz"]
_C.NET_DATALOADER.IMAGE_DIR = '/home2/migrate_handan/Biobank_images/'#'../data/'
_C.NET_DATALOADER.TARGET_DICT = '../data/age_dict.npy'
## age range setting


_C.NET_DATALOADER.BIN_RANGE = [42, 82]
_C.NET_DATALOADER.BIN_STEP =  1
_C.NET_DATALOADER.SIGMA = 1

_C.NET_DATALOADER.NUM_WORKERS = 32
_C.NET_DATALOADER.BATCH_SIZE = 24

# set pin_memory to TRUE when GPU memory is rich
_C.NET_DATALOADER.PIN_MEMORY = False
#
_C.NET_DATALOADER.DROP_LAST = True

_C.NET_DATALOADER.TEST_BATCH_SIZE = 1
_C.NET_DATALOADER.TEST_IMAGE_DIR = '../data/TEST/'

_C.NET_DATALOADER.DATA_AUG = True




# --------------------------------------------------------------
# Config of training
# --------------------------------------------------------------
_C.MSD_TRAIN = CN()
_C.MSD_TRAIN.USE_GPU = True#True
_C.MSD_TRAIN.DEVICE_ID = '0,1,2'#'0,1,2'
_C.MSD_TRAIN.DATA_DEVICE_ID = [0,1,2]#[0,1,2]
_C.MSD_TRAIN.FINE_TUNE = False
_C.MSD_TRAIN.PRETRAIN_MODEL_PATH = None
_C.MSD_TRAIN.EPOCHS = 100
_C.MSD_TRAIN.TEST_RATIO = 0.1

_C.MSD_TRAIN.VISDOM = False

_C.MSD_TRAIN.EARLY_STOPPING = True
_C.MSD_TRAIN.PATIENCE = 20
# --------------------------------------------------------------
# Config of SOLVER
# --------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.SEED = 1921
_C.SOLVER.LOSS = 'KL' # 'MSE' 'W-Distance' ''

_C.SOLVER.NAME = 'Adam' # 'Adam'

_C.SOLVER.SGD_LR = 0.01
_C.SOLVER.SGD_WEIGHT_DECAY = 0.001
_C.SOLVER.SGD_MOMENTUM = 0


_C.SOLVER.ADAM_LR = 0.001
_C.SOLVER.ADAM_BETAS = (0.9, 0.999)
_C.SOLVER.ADAM_EPS = 1e-08
_C.SOLVER.ADAM_WEIGHT_DECAY = 0#5*10**(-4)

if _C.SOLVER.NAME is 'SGD':
    _C.SOLVER.INIT_LR = _C.SOLVER.SGD_LR
elif _C.SOLVER.NAME is 'Adam':
    _C.SOLVER.INIT_LR = _C.SOLVER.ADAM_LR

_C.SOLVER.LR_DECAY = 0.1
_C.SOLVER.LR_PERIOD = 15

# --------------------------------------------------------------
# Config of TEST
# --------------------------------------------------------------
_C.MSD_TEST = CN()

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.NET_OUTPUT_DIR = "./msdmodel_add/"


# ---------------------------------------------------------------------------- #
# Visualization
# ---------------------------------------------------------------------------- #
_C.VISUALIZER = CN()
_C.VISUALIZER.ENV = '3DMSD'
