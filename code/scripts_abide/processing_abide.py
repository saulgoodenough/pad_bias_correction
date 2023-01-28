
import pandas as pd
import numpy as np



def process_age_abide():
    age_text_path = '../../../MRI_DATA/ABIDE/5320_ABIDE_Phenotypics_20211023.csv'
    df_age = pd.read_csv(age_text_path, sep=',', index_col=0, header=0, skiprows=[1])
    df_age.index = df_age.index.map(str)
    dict_age = df_age['AgeAtScan'].to_dict()
    np.save('../../../MRI_DATA/ABIDE/age_dict.npy', dict_age)
    dict_age_load = np.load('../../../MRI_DATA/ABIDE/age_dict.npy', allow_pickle=True).item()
    print(type(dict_age_load))
    print(dict_age_load['A00032016'])
    print(dict_age_load['A00032045'])
    return dict_age_load


def process_phenotype_abide(phenotype):
    data_path = '../../../MRI_DATA/ABIDE/5320_ABIDE_Phenotypics_20211023.csv'
    df_phenotype = pd.read_csv(data_path, sep=',', index_col=0, header=0, skiprows=[1])
    df_phenotype.index = df_phenotype.index.map(str)
    dict_phenotype = df_phenotype[phenotype].to_dict()
    np.save('../../../MRI_DATA/ABIDE/{}_dict.npy'.format(phenotype), dict_phenotype)
    dict_phenotype_load = np.load('../../../MRI_DATA/ABIDE/{}_dict.npy'.format(phenotype), allow_pickle=True).item()
    print(type(dict_phenotype_load))
    print(dict_phenotype_load['A00032016'])
    print(dict_phenotype_load['A00032045'])
    return dict_phenotype_load


def precess_age():
    age_text_path = '../../data/UKB_age_21003.txt'
    df_age = pd.read_csv(age_text_path, sep='\t', index_col=0, header= 0)
    print(df_age.head(5))
    print(df_age.iloc[0])

    df_age['age'] = np.nanmax(df_age.iloc[:, 1:].values, axis=1)
    print(df_age.head(5))
    print(df_age.iloc[0])

    print('the minimum age:')

    min_3_4 = np.array(df_age.iloc[:, 3:5].values)

    print(np.nanmin(min_3_4))

    #df_aprox_age = df_age.iloc[0:-1, 1:2]
    #print(df_aprox_age.head(5))
    df_age.index = df_age.index.map(str)
    dict_age = df_age['age'].to_dict()
    np.save('../../data/age_dict.npy', dict_age)
    dict_age_load = np.load('../../data/age_dict.npy', allow_pickle=True).item()

    #print(dict_age)
    print(type(dict_age_load))
    print(dict_age_load['2778480'])
    print(dict_age_load['4565682'])
    print(dict_age_load['3107973'])
    #print(df_aprox_age.loc[2778480,:])

def getid():
    path = '/home/tiger/Documents/multimodal/data/fmri/1001739/T1/T1.nii.gz'
    print(path.split('/')[-3])

from os import listdir
from os.path import join
import os
import glob

def is_mri_file(filename, endlist):
    return any(filename.endswith(extension) for extension in endlist)

def getallfiles(path):
    allfile = []
    for dirpath,dirnames,filenames in os.walk(path):
        for dir in dirnames:
            allfile.append(os.path.join(dirpath,dir))
            for name in filenames:
                allfile.append(os.path.join(dirpath, name))
    return allfile

def listfile():
    image_dir = '../../../RI_DATA/ABIDE/mri/'
    endlist = ['*MPRAGE.nii.gz']
    filepathlist = []
    for extension in endlist:
        filepathlist = filepathlist + glob.glob('/home1/zhangbiao/MRI_DATA/ABIDE/mri/**/'+extension, recursive=True)
    #filelist = [join(image_dir, x) for x in filepathlist if is_mri_file(x, endlist)]
    #print(filepathlist)
    #print(len(filepathlist))
    return filepathlist

import nibabel as nib
import numpy as np
import sys
sys.path.append('../')
import torch
from dp_model import dp_utils as dpu
import csv

def loadmri(mri_path):
    image_obj = nib.load(mri_path)
    image_data = image_obj.get_fdata()
    return image_data

import glob

def getfiles(cfg,  image_dir = None, filelist = None):
    #target_dict = np.load(target_dict_path, allow_pickle=True).item()
    #input_size = cfg.MSD_INPUT.CROP_SIZE
    if filelist is not None:
        image_filenames = filelist
    else:
        endlist = cfg.MSD_DATALOADER.ENDLIST
        image_dir = image_dir

        image_filenames = []
        for extension in endlist:
            image_filenames = image_filenames + glob.glob(image_dir + '**/' + extension, recursive=True)
    return image_filenames




def find_broken_file():
    '''
    path_list = [
        #'/home2/migrate_handan/Biobank_images/fmri_2/1361462/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri_2/3747581/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri/2642105/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri_2/4404824/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri_2/2135646/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri_2/4368033/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri_2/1720133/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri_2/5095598/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri_2/1307213/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri/5428846/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri/4964658/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri_2/1175069/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri/4973964/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri/2950851/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri/2030778/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri_2/1175069/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri_2/1003917/T1/T1.nii.gz',
        '/home2/migrate_handan/Biobank_images/fmri_2/4094422/T1/T1.nii.gz'
    ]
    input_size = (150, 192, 160)
    for path in path_list:
        input = loadmri(path)
        input = input / input.mean()
        # print(input.shape)
        input = dpu.crop_center(input, input_size)
        print(path)
        print(np.shape(input))
        sp_input = (1,) + input.shape
        input = input.reshape(sp_input)
        # sp_target = (1,) + target.shape
        # target = target.reshape(sp_target)

        input = torch.tensor(input, dtype=torch.float32)
    '''
    from config import default
    cfg = default._C
    image_filenames = getfiles(cfg, image_dir = cfg.MSD_DATALOADER.IMAGE_DIR)
    image_filenames.remove('/home2/migrate_handan/Biobank_images/fmri_2/1361462/T1/T1.nii.gz')

    image_filenames.remove('/home2/migrate_handan/Biobank_images/fmri_2/2154727/T1/T1.nii.gz')
    image_filenames.remove('/home2/migrate_handan/Biobank_images/fmri_2/1356383/T1/T1.nii.gz')


    #with open("mri_names.csv", "wb") as f:
    #    writer = csv.writer(f)
    #    writer.writerows(image_filenames)
    k = 0
    broken_list = []
    for path in image_filenames:
        print(path)
        input = loadmri(path)
        input = input / input.mean()
        # print(input.shape)
        input = dpu.crop_center(input, cfg.MSD_INPUT.CROP_SIZE)
        k += 1
        print(k)
        #print(np.shape(input))
        try:
            sp_input = (1,) + input.shape
            input = input.reshape(sp_input)
            # sp_target = (1,) + target.shape
            # target = target.reshape(sp_target)

            input = torch.tensor(input, dtype=torch.float32)

            print(f'The {k} well-downloaded file...')
            print(path)

        except EOFError:
            print('Broken file:', path)
            broken_list.append(path)
            #continue

    print('----------------Broken files-----------------')
    print(broken_list)

if __name__ == '__main__':
    '''
    data = np.random.rand(159, 220, 150)
    out_sp = (160, 200, 160)
    output = dpu.crop_center(data, out_sp)
    print(output.shape)

    data = np.random.rand(1, 2, 2)
    print(data)
    out_sp = (2, 2, 2)
    output = dpu.crop_center(data, out_sp)
    print(output.shape)
    print(output)
    '''
    #process_age_abide()
    #listfile()
    '''
    mri_image = loadmri('/home1/zhangbiao/MRI_DATA/ABIDE/mri/scan_data001/um/dicom/signa/mmilham/abide_28730/A00032419/394504574_session_1/mprage_0001/MPRAGE.nii.gz')
    print(mri_image.shape)

    age_dict = process_age_abide()
    age_list = list(age_dict.values())
    print(len(age_list))
    print('max age:', max(age_list))
    print('key:', list(age_dict.keys())[list(age_dict.values()).index(max(age_list))])
    print('min age:', min(age_list))
    print('key:', list(age_dict.keys())[list(age_dict.values()).index(min(age_list))])

    filepathlist = listfile()
    print(len(filepathlist))

    age_mean = np.mean(np.array(age_list))
    age_std = np.std(np.array(age_list))
    print('age mean:', age_mean)
    print('age std:', age_std)
    '''
    phenotype_list = ['BMI', 'FIQ', 'VIQ', 'PIQ', 'Handedness_Scores', 'SRS_awareness', 'SRS_cognition', 'SRS_communication']
    for phenotype in phenotype_list:
        process_phenotype_abide(phenotype)
