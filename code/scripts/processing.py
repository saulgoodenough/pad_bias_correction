
import pandas as pd
import numpy as np

def precess_age():
    age_text_path = '../../../data/UKB_age_21003.txt'
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
    image_dir = '/home2/migrate_handan/Biobank_images/'
    endlist = ['T1.nii.gz']
    filepathlist = []
    for extension in endlist:
        filepathlist = filepathlist + glob.glob(image_dir+ '**/'+extension, recursive=True)
    #filelist = [join(image_dir, x) for x in filepathlist if is_mri_file(x, endlist)]
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

    '''
    #precess_age()
    age_dict = dict_age_load = np.load('../../data/age_dict.npy', allow_pickle=True).item()
    age_list = []
    filelist = listfile()
    for mripath in filelist:
        userid = mripath.split('/')[-3]
        target_age = age_dict[userid]
        age_list.append(target_age)

    age_mean = np.mean(np.array(age_list))
    age_std = np.std(np.array(age_list))
    print('age mean:', age_mean)
    print('age std:', age_std)
    '''

    raw_data_path = '/home2/UKB_Tabular_merged_10/'
    UKB_fieldid_subset_df = pd.read_csv('/home2/UKB_Tabular_merged_10/UKB_FieldID_Subset.csv', index_col=0)
    index_field_id_list = ['4125-2.0', '4080-0.1', '21002-2.0', '4079-0.1', '23236-2.0', '21001-2.0', '49-2.0', '20116-0.0', '30040-0.0',
                           '12702-2.0', '12687-2.1', '23235-2.0', '23120-0.0', '30050-0.0', '12673-2.0', '12682-2.0', '102-2.0', '23226-2.0',
                           '1558-2.0', '20015-2.0', '30010-0.0', '23105-0.0', '137-2.0', '2443-2.0', '1249-2.0', '30270-0.0', '23099-0.0', '20016-2.0',
                           '404-2.7', '20195-0.0', '20159-0.0', '3064-2.1', '22408-2.0', '1458-2.0', '20023-2.0', '1970-0.0', '738-2.0', '20133-0.1', '709-0.0',
                           '1588-0.0', '1568-0.0']
    index_name_list = ['Heel bone mineral density', 'Systolic blood pressure', 'Weight', 'Diastolic blood pressure', 'Total BMD (bone mineral density)',
                       'Body mass index (BMI)', 'Hip circumference', 'Still smoking', 'Mean corpuscular volume', 'Cardiac index during PWA',
                       'Mean srterial pressure during PWA', 'Total BMC (bone mineral content)', 'Arm fat mass (right)', 'Mean corpuscular haemoglobibin',
                       'Heart rate during PWA', 'Cardiac output during PWA', 'Pulse rate', 'Head BMD (bone mineral density)', 'Alcohol intake frequency',
                       'Sitting height', 'Red blood cell (erythrocyte) count', 'Basal metabolic rate', 'Number of treatments/medications taken', 'Diabetes diagnosed by doctor',
                       'Past tobacco smoking', 'Mean sphered cell volume', 'Duration to complete alphanumeric path (cognition)', 'Body fat percentage',
                       'Fluid intelligence score (cognition)', 'Duration to first press of snap-button (cognition)', 'Number of symbol digit matches attempted (cognition)',
                       'Number of symbol digit matches made correctly (cognition)', 'Peak expiratory flow (PEF)', 'Abdominal subcutaneous adipose tissue volume',
                       'Cereal intake', 'Mean time to correctly identify matches (cognition)', 'Nervous feelings', 'Average total household income before tax',
                       'Time to complete round, pairs matching (cognition)', 'Number in household', 'Average weekly beer plus cider intake', 'Average weekly red wine intake']


    df_index_subset = UKB_fieldid_subset_df.loc[index_field_id_list]
    data_save_path = '../../../MRI_DATA/UK_Biobank/'
    '''
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    df_index_subset.to_csv(data_save_path + 'index_subset.csv')

    file_group = df_index_subset.groupby('Subset_ID', as_index=False)
    print(file_group)
    for key, group in file_group:
        print('Group')
        print(key, group)
        print('Group Subset_ID')
        print(key, group['Subset_ID'])
        subset_file_name = raw_data_path + 'UKB_subset_{}.csv'.format(int(group['Subset_ID'][0]))
        print(['eid']+list(group.index))
        subset_df = pd.read_csv(subset_file_name, low_memory=False)[['eid']+list(group.index)]
        subset_df.set_index('eid')
        subset_df.to_csv(data_save_path + 'subset_{}.csv'.format(int(key)))
        
    
    df_1 = pd.read_csv(data_save_path + 'subset_1.csv')
    df_1 = df_1.set_index('eid')
    df_1.drop(columns=df_1.columns[0], axis=1, inplace=True)
    df_4 = pd.read_csv(data_save_path + 'subset_4.csv')
    df_4 = df_4.set_index('eid')
    df_4.drop(columns=df_4.columns[0], axis=1, inplace=True)
    df_5 = pd.read_csv(data_save_path + 'subset_5.csv')
    df_5 = df_5.set_index('eid')
    df_5.drop(columns=df_5.columns[0], axis=1, inplace=True)
    df_8 = pd.read_csv(data_save_path + 'subset_8.csv')
    df_8 = df_8.set_index('eid')
    df_8.drop(columns=df_8.columns[0], axis=1, inplace=True)
    df_all = pd.concat([df_1, df_4, df_5, df_8], axis=1)
    df_all.to_csv(data_save_path + 'index_used.csv')
    '''

    df_all = pd.read_csv(data_save_path + 'index_used.csv')
    df_age_index = df_all.apply(lambda row: str((str(int(row['eid'])),)), axis=1)
    df_all.index = df_age_index
    df_all.to_csv(data_save_path + 'index_used_newid.csv')






