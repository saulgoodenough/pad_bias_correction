
import pandas as pd
import numpy as np

def process_age_abide():
    age_text_path = '../../../MRI_DATA/ABIDE/5320_ABIDE_Phenotypics_20211023.csv'
    df_age = pd.read_csv(age_text_path, sep=',', index_col=0, header=0)
    df_age.index = df_age.index.map(str)
    dict_age = df_age['AgeAtScan'].to_dict()
    np.save('../../../MRI_DATA/ABIDE/age_dict.npy', dict_age)
    dict_age_load = np.load('../../../MRI_DATA/ABIDE/age_dict.npy', allow_pickle=True).item()
    print(type(dict_age_load))
    print(dict_age_load['A00032016'])
    print(dict_age_load['A00032045'])

def process_age_oasis():
    age_text_path = '../../../MRI_DATA/oasis/oasis_agedata.csv'
    df_age = pd.read_csv(age_text_path, sep=',', index_col=0, header=0)
    df_age.index = df_age['Subject'].map(str)
    dict_age = df_age['ageAtEntry'].to_dict()
    np.save('../../../MRI_DATA/oasis/age_dict_raw.npy', dict_age)
    dict_age_load = np.load('../../../MRI_DATA/oasis/age_dict_raw.npy', allow_pickle=True).item()
    print(type(dict_age_load))
    print(dict_age_load['OAS30124'])
    print(dict_age_load['OAS30354'])
    #return dict_age

def process_oasis_to_csv():
    #name = 'CortexVol'
    age_text_path = '../../../MRI_DATA/oasis/oasis_brain_volume.csv'
    df_age = pd.read_csv(age_text_path, sep=',', index_col=None, header=0)
    print(df_age)
    #df_age.index = df_age['Subject'].map(str)
    #df_age = df_age.groupby(['Subject']).mean() "('sub-OAS30154_ses-d0244_T1w.nii.gz',)"
    # "('sub-OAS31047_ses-d0183_run-02_T1w.nii.gz',)"

    df_age_index = df_age.apply(lambda row: str(('sub-'+ row['FS_FSDATA ID'].split('_')[0] + '_ses' + '-' +
                                            row['FS_FSDATA ID'].split('_')[2] + '_run-'
                                            + row['Session'].split('_')[0].strip('CENTRAL') + '_T1w.nii.gz' if '0' in row['Session'].split('_')[0]
                                            else 'sub-'+ row['FS_FSDATA ID'].split('_')[0] + '_ses' + '-' +
                                            row['FS_FSDATA ID'].split('_')[2] + '_T1w.nii.gz',)), axis=1)

    df_age.index = df_age_index
    df_age.to_csv('../../../MRI_DATA/oasis/oasis_brain_volume_newid.csv')
    #df_age.index = df_age['Subject'].map(str)

    '''
    dict_output = df_age[name].to_dict()
    np.save('../../../MRI_DATA/oasis/{}_dict_raw.npy'.format(name), dict_output)
    dict_age_load = np.load('../../../MRI_DATA/oasis/{}_dict_raw.npy'.format(name), allow_pickle=True).item()
    print(type(dict_age_load))
    print(dict_age_load)
    '''

    whole_path = '../../oasis_dataset/predict/resnet3d34/2021-12-02_age_predict.csv'
    df_predict = pd.read_csv(whole_path , sep=',', index_col=0, header=0)

    print(df_predict)
    print(df_predict.index)

    # common samples
    list_intersect = list(set(df_predict.index) & set(df_age.index))
    print(len(list_intersect))







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
    image_dir = '../../../MRI_DATA/ABIDE/mri/'
    endlist = ['*MPRAGE.nii.gz']
    filepathlist = []
    for extension in endlist:
        filepathlist = filepathlist + glob.glob('/home1/zhangbiao/MRI_DATA/ABIDE/mri/**/'+extension, recursive=True)
    #filelist = [join(image_dir, x) for x in filepathlist if is_mri_file(x, endlist)]
    print(filepathlist)
    print(len(filepathlist))


def listfile_oasis():
    image_dir = '../../../MRI_DATA/oasis/mri/'
    endlist = ['*T1w.nii.gz']
    filepathlist = []
    for extension in endlist:
        filepathlist = filepathlist + glob.glob(image_dir+'**/'+extension, recursive=True)
    #filelist = [join(image_dir, x) for x in filepathlist if is_mri_file(x, endlist)]
    #print(filepathlist)
    print(len(filepathlist))
    return filepathlist

def rewrite_age_dict_oasis(age_dict, filepathlist):
    age_dict_new = {}
    print(age_dict)
    print(len(age_dict.keys()))
    for mrifile in filepathlist:
        index_subject =  mrifile.split('/')[-4]
        index_subject_list = index_subject.split('_')
        index_subject = index_subject_list[0]
        #print(index)
        index_mri = mrifile.split('/')[-1]
        #try:
        #print(index_subject, index_mri)

        if index_subject in age_dict.keys():
            print(index_subject)
            print(age_dict[index_subject])
            age_dict_new[index_mri] = age_dict[index_subject]
        #except:
        #    continue
    print(list(age_dict_new.keys())[0:10])
    print(len(age_dict_new.keys()))

    print('sub-OAS30480_ses-d0161_run-01_T1w.nii.gz', age_dict_new['sub-OAS30480_ses-d0161_run-01_T1w.nii.gz'])
    # 74.157425
    np.save('../../../MRI_DATA/oasis/age_dict.npy', age_dict_new)
    return age_dict_new

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

    #mri_image = loadmri('/home1/zhangbiao/MRI_DATA/ABIDE/mri/scan_data001/um/dicom/signa/mmilham/abide_28730/A00032419/394504574_session_1/mprage_0001/MPRAGE.nii.gz')
    #print(mri_image.shape)
    #process_age_oasis()

    #mri_image = loadmri('/home1/zhangbiao/MRI_DATA/oasis/mri/OAS30005_MR_d0143/anat3/NIFTI/sub-OAS30005_ses-d0143_T1w.nii.gz')
    #print(mri_image.shape)

    #process_age_oasis()
    '''
    dict_age_load = np.load('../../../MRI_DATA/oasis/age_dict_raw.npy', allow_pickle=True).item()
    print(len(dict_age_load.keys()))
    filelists = listfile_oasis()
    print(len(filelists))
    age_dict = rewrite_age_dict_oasis(dict_age_load, filelists)

    age_all_list = list(age_dict.values())
    age_mean = np.mean(np.array(age_all_list))
    age_std = np.std(np.array(age_all_list))
    print('age mean:', age_mean)
    print('age std:', age_std)

    age_list = list(dict_age_load.values())
    print(max(age_list), min(age_list))
    '''
    process_oasis_to_csv()

