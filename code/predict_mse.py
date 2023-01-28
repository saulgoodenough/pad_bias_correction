import os
from torch.backends import cudnn
from utils.logger import setup_logger
import time

import random
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import numpy as np
import os
import argparse

from utils.loss import age_diff_mse
from utils.loss import age_diff

from utils.data_utils import DatasetFromFolder_mri_leftright
from utils.data_utils import count_parameters

from utils.solver_utils import makeCritirion
from utils.solver_utils import makeSolver

#import msd3d
#from msd3d import MSD3D

#import sys
#sys.path.append('../')

from dp_model.model_files import sfcn_leftright as sfcn_leftright
from dp_model.model_files import sfcn_mse as sfcn

from dp_model.model_files import resnet3d_mse as resnet3d
from dp_model.model_files import resnet3d_leftright as resnet3d_leftright

from dp_model.model_files import msd3d_mse as msd3d

from Visualizer import Visualizer

from torch.cuda.amp import GradScaler

from utils.pytorchtools import EarlyStopping

import csv

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def predict(cfg, trained_model_dict):
    cfg.freeze()

    # setting seed
    if cfg.NET_TRAIN.USE_GPU:
        set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.NET_PREDICT.WHOLE_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("3d net model", output_dir, if_train=False)
    logger.info("Saving model in the path :{}".format(cfg.NET_TRAIN.NET_OUTPUT_DIR))

    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.NET_TRAIN.DEVICE_ID

    # GPU accelerate
    if cfg.NET_TRAIN.USE_GPU:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        #torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')

    logger.info('===> Loading training datasets')

    image_file_list = list(np.load(cfg.NET_DATALOADER.TEST_IMAGELIST_PATH, allow_pickle=True))
    #print(image_file_list)
    test_set = DatasetFromFolder_mri_leftright(cfg=cfg, target_dict_path=cfg.NET_DATALOADER.TARGET_DICT, filelist=image_file_list,  direction=cfg.NET_DATALOADER.DIRECTION, mode = 'test')
    print(len(test_set))
    test_data_loader = DataLoader(dataset=test_set, num_workers=cfg.NET_DATALOADER.NUM_WORKERS,
                                      batch_size=cfg.NET_DATALOADER.TEST_BATCH_SIZE, shuffle=False,
                                      pin_memory=cfg.NET_DATALOADER.PIN_MEMORY, drop_last=cfg.NET_DATALOADER.DROP_LAST)


    logger.info('===> Loading testing datasets')

    if cfg.NET_TRAIN.TRAIN_TYPE == 'whole':
        # --------------------------------------------------------------------------------------------------
        logger.info('===> Building whole model')

        if cfg.NET_MODEL.TYPE == 'SFCN':
            model = sfcn.SFCN(output_dim=cfg.NET_MODEL.CLASS_NUM)
            # exec_line = 'net_output = Net(input)[0]\ntarget = target.reshape([net_output.size()[0], net_output.size()[1], 1, 1, 1])'

        elif cfg.NET_MODEL.TYPE == 'RESNET3D18':
            model = resnet3d.resnet18(num_classes=cfg.NET_MODEL.CLASS_NUM)
            # exec_line = 'net_output = Net(input)\ntarget = target.reshape([net_output.size()[0], net_output.size()[1]])'
        elif cfg.NET_MODEL.TYPE == 'RESNET3D34':
            model = resnet3d.resnet34(num_classes=cfg.NET_MODEL.CLASS_NUM)
        elif cfg.NET_MODEL.TYPE == 'RESNET3D50':
            model = resnet3d.resnet34(num_classes=cfg.NET_MODEL.CLASS_NUM)
        elif cfg.NET_MODEL.TYPE == 'MSD3D':
            model = msd3d.MSD3D(cfg)
        # model = MSD3D(cfg)
        Net = nn.DataParallel(model, device_ids=cfg.NET_TRAIN.DATA_DEVICE_ID)

        para = count_parameters(model)
        logger.info(f'Amount of parameters:  {para}')

        criterion = makeCritirion(cfg.SOLVER.LOSS)  # nn.NLLLoss()  # nn.MSELoss() ## nn.CrossEntropyLoss()##
        if cfg.NET_TRAIN.USE_GPU:
            Net = Net.cuda()

        # Waiting for the end of all the center program
        if cfg.NET_TRAIN.USE_GPU:
            torch.cuda.synchronize()

        time_start = time.time()

        # visualization
        if cfg.NET_TRAIN.VISDOM:
            vis = Visualizer(env=cfg.VISUALIZER.ENV)

        logger.info('===> start training')

        age_range = torch.Tensor(cfg.NET_DATALOADER.BIN_RANGE)

        bc = np.arange(cfg.NET_DATALOADER.BIN_RANGE[0], cfg.NET_DATALOADER.BIN_RANGE[1]) + 0.5
        bc = torch.tensor(bc, dtype=torch.float32).to(device)

        # left predition---------------------------------------------------------------------------------------

        print('----------------WHOLE prediction------------------------------------------------------')
        #Net.load_state_dict(torch.load(cfg.NET_PREDICT.WHOLE_MODEL))
        if cfg.NET_PREDICT.WHOLE_DIR and not os.path.exists(cfg.NET_PREDICT.WHOLE_DIR):
            os.makedirs(cfg.NET_PREDICT.WHOLE_DIR)
        Net.load_state_dict(torch.load(trained_model_dict['whole']))

        # predict the test set
        logger.info('===> start test predition')
        torch.cuda.empty_cache()
        test_epoch_loss = 0
        age_diff_loss = 0

        # write age prediction
        predict_file = cfg.NET_PREDICT.WHOLE_DIR + time.strftime("%Y-%m-%d") + '_age_predict.csv'
        predict_file = open(predict_file, "w")  # 创建csv文件
        writer = csv.writer(predict_file)
        writer.writerow(['user id', 'predict age', 'real age', 'age difference'])
        with torch.no_grad():
            for iteration, batch in enumerate(test_data_loader, 1):
                input = batch[5].to(device)
                target = batch[2].to(device)
                userid = batch[3]
                target_age = batch[4].to(device)

                output = Net(input)
                test_loss = criterion(output, target_age)
                age_x, age_difference = age_diff_mse(output, target_age)
                test_epoch_loss += test_loss.item()
                age_diff_loss += age_difference.item()
                writer.writerow([userid, age_x.item(), target_age.item(), age_difference.item()])
                # print("===> Epoch[{}]({}/{}):Validation Batch Loss: {:.8f}".format(epoch, iteration, len(testing_data_loader),  test_loss.item()))
        avg_test_loss = test_epoch_loss / len(test_data_loader)
        ave_age_loss = age_diff_loss / len(test_data_loader)
        logger.info("WHOLE ===>  Avg. Validation Loss: {:.8f}".format(avg_test_loss))
        logger.info("WHOLE ===> Avg. Validation age difference: {:.8f}".format(ave_age_loss))

        predict_file.close()

        if cfg.NET_TRAIN.USE_GPU:
            torch.cuda.synchronize()
        time_elapsed = time.time() - time_start
        logger.info('WHOLE test elapse  {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # right predition---------------------------------------------------------------------------------------


    elif cfg.NET_TRAIN.TRAIN_TYPE == 'left_right':
        # --------------------------------------------------------------------------------------------------
        logger.info('===> Building left model')

        if cfg.NET_MODEL.TYPE == 'SFCN':
            model = sfcn_leftright.SFCN(output_dim=cfg.NET_MODEL.CLASS_NUM)
            # exec_line = 'net_output = Net(input)[0]\ntarget = target.reshape([net_output.size()[0], net_output.size()[1], 1, 1, 1])'
        elif cfg.NET_MODEL.TYPE == 'RESNET3D18':
            model = resnet3d_leftright.resnet18(num_classes=cfg.NET_MODEL.CLASS_NUM)
            # exec_line = 'net_output = Net(input)\ntarget = target.reshape([net_output.size()[0], net_output.size()[1]])'
        elif cfg.NET_MODEL.TYPE == 'RESNET3D34':
            model = resnet3d_leftright.resnet34(num_classes=cfg.NET_MODEL.CLASS_NUM)
        elif cfg.NET_MODEL.TYPE == 'RESNET3D50':
            model = resnet3d_leftright.resnet34(num_classes=cfg.NET_MODEL.CLASS_NUM)
        elif cfg.NET_MODEL.TYPE == 'MSD3D':
            model = msd3d.MSD3D(cfg)
        # model = MSD3D(cfg)
        Net = nn.DataParallel(model, device_ids=cfg.NET_TRAIN.DATA_DEVICE_ID)

        para = count_parameters(model)
        logger.info(f'Amount of parameters:  {para}')

        criterion = makeCritirion(cfg.SOLVER.LOSS)  # nn.NLLLoss()  # nn.MSELoss() ## nn.CrossEntropyLoss()##
        if cfg.NET_TRAIN.USE_GPU:
            Net = Net.cuda()

        # Waiting for the end of all the center program
        if cfg.NET_TRAIN.USE_GPU:
            torch.cuda.synchronize()

        time_start = time.time()

        # visualization
        if cfg.NET_TRAIN.VISDOM:
            vis = Visualizer(env=cfg.VISUALIZER.ENV)

        logger.info('===> start training')

        age_range = torch.Tensor(cfg.NET_DATALOADER.BIN_RANGE)

        bc = np.arange(cfg.NET_DATALOADER.BIN_RANGE[0], cfg.NET_DATALOADER.BIN_RANGE[1]) + 0.5
        bc = torch.tensor(bc, dtype=torch.float32).to(device)

    # left predition---------------------------------------------------------------------------------------

        print('----------------left prediction------------------------------------------------------')
        #Net.load_state_dict(torch.load(cfg.NET_PREDICT.LEFT_MODEL))
        Net.load_state_dict(torch.load(trained_model_dict['left']))
        if cfg.NET_PREDICT.LEFT_DIR and not os.path.exists(cfg.NET_PREDICT.LEFT_DIR):
            os.makedirs(cfg.NET_PREDICT.LEFT_DIR)

        # predict the test set
        logger.info('===> start test predition')
        torch.cuda.empty_cache()
        test_epoch_loss = 0
        age_diff_loss = 0

        # write age prediction
        predict_file = cfg.NET_PREDICT.LEFT_DIR + time.strftime("%Y-%m-%d") +'_age_predict.csv'
        predict_file = open(predict_file, "w")  # 创建csv文件
        writer = csv.writer(predict_file)
        writer.writerow(['user id', 'predict age', 'real age', 'age difference'])
        with torch.no_grad():
            for iteration, batch in enumerate(test_data_loader, 1):
                input = batch[0].to(device)
                target = batch[2].to(device)
                userid = batch[3]
                target_age = batch[4].to(device)

                output = Net(input)[0]
                test_loss = criterion(output, target)
                age_x, age_difference = age_diff(output, target_age, bc)
                test_epoch_loss += test_loss.item()
                age_diff_loss += age_difference.item()
                writer.writerow([userid, age_x.item(), target_age.item(), age_difference.item()])
                # print("===> Epoch[{}]({}/{}):Validation Batch Loss: {:.8f}".format(epoch, iteration, len(testing_data_loader),  test_loss.item()))
        avg_test_loss = test_epoch_loss / len(test_data_loader)
        ave_age_loss = age_diff_loss / len(test_data_loader)
        logger.info("LEFT ===>  Avg. Validation Loss: {:.8f}".format(avg_test_loss))
        logger.info("LEFT ===> Avg. Validation age difference: {:.8f}".format(ave_age_loss))

        predict_file.close()

        if cfg.NET_TRAIN.USE_GPU:
            torch.cuda.synchronize()
        time_elapsed = time.time() - time_start
        logger.info('LEFT test elapse  {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # right predition---------------------------------------------------------------------------------------

        print('----------------right prediction------------------------------------------------------')
        #Net.load_state_dict(torch.load(cfg.NET_PREDICT.RIGHT_MODEL))
        Net.load_state_dict(torch.load(trained_model_dict['right']))
        if cfg.NET_PREDICT.RIGHT_DIR and not os.path.exists(cfg.NET_PREDICT.RIGHT_DIR):
            os.makedirs(cfg.NET_PREDICT.RIGHT_DIR)
        # predict the test set
        logger.info('===> start test predition')
        torch.cuda.empty_cache()
        test_epoch_loss = 0
        age_diff_loss = 0

        # write age prediction
        predict_file = cfg.NET_PREDICT.RIGHT_DIR + time.strftime("%Y-%m-%d") + '_age_predict.csv'
        predict_file = open(predict_file, "w")  # 创建csv文件
        writer = csv.writer(predict_file)
        writer.writerow(['user id', 'predict age', 'real age', 'age difference'])
        with torch.no_grad():
            for iteration, batch in enumerate(test_data_loader, 1):
                input = batch[1].to(device)
                target = batch[2].to(device)
                userid = batch[3]
                target_age = batch[4].to(device)

                output = Net(input)[0]
                test_loss = criterion(output, target)
                age_x, age_difference = age_diff(output, target_age, bc)
                test_epoch_loss += test_loss.item()
                age_diff_loss += age_difference.item()
                writer.writerow([userid, age_x.item(), target_age.item(), age_difference.item()])
                # print("===> Epoch[{}]({}/{}):Validation Batch Loss: {:.8f}".format(epoch, iteration, len(testing_data_loader),  test_loss.item()))
        avg_test_loss = test_epoch_loss / len(test_data_loader)
        ave_age_loss = age_diff_loss / len(test_data_loader)
        logger.info("RIGHT ===>  Avg. Validation Loss: {:.8f}".format(avg_test_loss))
        logger.info("RIGHT ===> Avg. Validation age difference: {:.8f}".format(ave_age_loss))

        predict_file.close()

        if cfg.NET_TRAIN.USE_GPU:
            torch.cuda.synchronize()
        time_elapsed = time.time() - time_start
        logger.info('LEFT test elapse  {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
























