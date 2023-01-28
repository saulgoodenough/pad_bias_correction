from yacs.config import CfgNode as CN


# --------------------------------------------------------------
'''
_C.NET3D: Hyper paramters for general net

'''
# --------------------------------------------------------------
# Config of model
# --------------------------------------------------------------
_C = CN()

_C.PCA = CN()

_C.PCA.CLASS_NUM = 48 #40
_C.PCA.DIMENSION = 2000 #40
_C.PCA.IMAGE_SIZE = (162, 226, 209)
_C.PCA.CROP_SIZE = (160, 192, 160)
_C.PCA.FILTERED_SIZE = (4, 4, 4)
_C.PCA.ENDLIST = ["T1.nii.gz"]
_C.PCA.IMAGE_DIR = '/home2/migrate_handan/Biobank_images/'#'../data/'
_C.PCA.TARGET_DICT = '../../data/age_dict.npy'

_C.PCA.MODEL_SAVE_PATH = '../../data/pca_model_sparse.sav'
_C.PCA.DATA_SAVE_PATH = '../../data/pca_ukbiobank_sparse.npz'

_C.PCA.SPARSE_SAVE_PATH = '../../data/ukbiobank_sparse_mat.npz'


_C.PCA.COMBINE_DATA_SAVE_PATH = '../../data/pca_combine_sparse.npz'


_C.PCA.TRAIN_IMAGELIST_PATH = '../../data/image_list_train.npy'
_C.PCA.TEST_IMAGELIST_PATH = '../../data/image_list_test.npy'
_C.PCA.TEST_PERCENT = 0.05

## age range setting
_C.PCA.BIN_RANGE = [38, 86]#[42, 82]

_C.PCA.NUM_WORKERS = 32


