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

from utils.data_utils import DatasetFromFolder_mri_leftright
from utils.data_utils import count_parameters
from utils.data_utils import ImbalancedDatasetSampler


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

import pandas as pd


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def trainmodel_mse(cfg):
    cfg.freeze()

    trained_model_dict = {'whole': '', 'left': '', 'right': ''}

    # setting seed
    if cfg.NET_TRAIN.USE_GPU:
        set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.NET_TRAIN.NET_OUTPUT_DIR

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if cfg.NET_TRAIN.WHOLE_NET_OUTPUT_DIR and not os.path.exists(cfg.NET_TRAIN.WHOLE_NET_OUTPUT_DIR):
        os.makedirs(cfg.NET_TRAIN.WHOLE_NET_OUTPUT_DIR)


    logger = setup_logger("3d net model", output_dir, if_train=True)
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


    image_file_list = list(np.load(cfg.NET_DATALOADER.TRAIN_IMAGELIST_PATH, allow_pickle=True))
    training_set = DatasetFromFolder_mri_leftright(cfg=cfg, image_dir=cfg.NET_DATALOADER.IMAGE_DIR, target_dict_path=cfg.NET_DATALOADER.TARGET_DICT, filelist=image_file_list, direction=cfg.NET_DATALOADER.DIRECTION, mode='train_notsave')
    print('Sample number:', len(training_set))
    train_set, validate_set = torch.utils.data.random_split(training_set,
                                                        [int(len(training_set) - int(cfg.NET_TRAIN.TEST_RATIO * len(training_set))),
                                                                    int(cfg.NET_TRAIN.TEST_RATIO * len(training_set))])
    #print(train_set.shape)


    if cfg.NET_TRAIN.BALANCE:
        print('Generating train sampler...')
        train_sampler = ImbalancedDatasetSampler(train_set)
        print('Generating train sampler finished!')
        training_data_loader = DataLoader(dataset=train_set, sampler=train_sampler, num_workers=cfg.NET_DATALOADER.NUM_WORKERS,
                                          batch_size=cfg.NET_DATALOADER.BATCH_SIZE, #shuffle=True,
                                          pin_memory=cfg.NET_DATALOADER.PIN_MEMORY,
                                          drop_last=cfg.NET_DATALOADER.DROP_LAST)
    else:
        training_data_loader = DataLoader(dataset=train_set,  num_workers=cfg.NET_DATALOADER.NUM_WORKERS,
                                      batch_size=cfg.NET_DATALOADER.BATCH_SIZE, shuffle=True,
                                      pin_memory=cfg.NET_DATALOADER.PIN_MEMORY, drop_last=cfg.NET_DATALOADER.DROP_LAST)


    logger.info('===> Loading testing datasets')

    # test_size = test_batch_num * batch_size  # test set size
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=cfg.NET_DATALOADER.NUM_WORKERS,
                                      batch_size=cfg.NET_DATALOADER.TEST_BATCH_SIZE, shuffle=True,
                                      pin_memory=cfg.NET_DATALOADER.PIN_MEMORY, drop_last=cfg.NET_DATALOADER.DROP_LAST)

    #net_output = 0



    if cfg.NET_TRAIN.TRAIN_TYPE == 'whole':
        # --------------------------------------------------------------------------------------------------

        logger.info('===> Building whole model')

        if cfg.NET_MODEL.TYPE == 'SFCN':
            model = sfcn.SFCN(output_dim=cfg.NET_MODEL.CLASS_NUM)
            #exec_line = 'net_output = Net(input)[0]\ntarget = target.reshape([net_output.size()[0], net_output.size()[1], 1, 1, 1])'

        elif cfg.NET_MODEL.TYPE == 'RESNET3D18':
            model = resnet3d.resnet18(num_classes=cfg.NET_MODEL.CLASS_NUM, bin_range= cfg.NET_DATALOADER.BIN_RANGE)
            #exec_line = 'net_output = Net(input)\ntarget = target.reshape([net_output.size()[0], net_output.size()[1]])'
        elif cfg.NET_MODEL.TYPE == 'RESNET3D34':
            model = resnet3d.resnet34(num_classes=cfg.NET_MODEL.CLASS_NUM, bin_range= cfg.NET_DATALOADER.BIN_RANGE)
        elif cfg.NET_MODEL.TYPE == 'RESNET3D50':
            model = resnet3d.resnet34(num_classes=cfg.NET_MODEL.CLASS_NUM, bin_range= cfg.NET_DATALOADER.BIN_RANGE)
        elif cfg.NET_MODEL.TYPE == 'MSD3D':
            model = msd3d.MSD3D(cfg)
        # model = MSD3D(cfg)
        Net = nn.DataParallel(model, device_ids=cfg.NET_TRAIN.DATA_DEVICE_ID)

        para = count_parameters(model)
        logger.info(f'Amount of parameters:  {para}')

        loss_type = cfg.SOLVER.LOSS
        criterion = makeCritirion(cfg.SOLVER.LOSS)  # nn.NLLLoss()  # nn.MSELoss() ## nn.CrossEntropyLoss()##
        if cfg.NET_TRAIN.USE_GPU:
            Net = Net.cuda()
        # set optimizer
        optimizer = makeSolver(cfg, Net)

        # Waiting for the end of all the center program
        if cfg.NET_TRAIN.USE_GPU:
            torch.cuda.synchronize()

        time_start = time.time()

        # visualization
        if cfg.NET_TRAIN.VISDOM:
            vis = Visualizer(env=cfg.VISUALIZER.WHOLE_ENV)

        # Auto Mixed Precision Training
        scaler = GradScaler()

        logger.info('===> start training')

        # init learning rate
        lr = cfg.SOLVER.INIT_LR

        # early stopping
        if cfg.NET_TRAIN.EARLY_STOPPING is not None:
            early_stopping = EarlyStopping(patience=cfg.NET_TRAIN.PATIENCE, verbose=True)

        age_range = torch.Tensor(cfg.NET_DATALOADER.BIN_RANGE)

        bc = np.arange(cfg.NET_DATALOADER.BIN_RANGE[0], cfg.NET_DATALOADER.BIN_RANGE[1]) + 0.5
        bc = torch.tensor(bc, dtype=torch.float32).to(device)

        # start training if trainig is interrupted
        epoch_append = 0
        if cfg.NET_TRAIN.FINE_TUNE:
            Net.load_state_dict(torch.load(cfg.NET_TRAIN.WHOLE_PRETRAIN_MODEL_PATH))
            epoch_append = int(cfg.NET_TRAIN.WHOLE_PRETRAIN_MODEL_PATH.split('_')[3])

        validate_last_loss = 100000



        for epoch in range(1 + epoch_append, cfg.NET_TRAIN.EPOCHS + 1 + epoch_append):
            epoch_loss = 0
            epoch_age_loss = 0
            if epoch > 0 and epoch % cfg.SOLVER.LR_PERIOD == 0 and lr > 1e-8:  # 每lr_period个epoch，学习率衰减一次
                lr = lr * cfg.SOLVER.LR_DECAY
                for param_group in optimizer.param_groups[6:]:
                    param_group['lr'] = lr
            for iteration, batch in enumerate(training_data_loader, 1):
                input = batch[5].to(device)
                target = batch[2].to(device)
                userid = batch[3]
                target_age = batch[4].to(device)
                # print('target_age', target_age)
                optimizer.zero_grad()
                # Casts operations to mixed precision
                # with torch.cuda.amp.autocast():

                '''
                if cfg.NET_MODEL.TYPE == 'SFCN':
                    net_output = Net(input)[0]
                    target = target.reshape([net_output.size()[0], net_output.size()[1], 1, 1, 1])
                else:
                    net_output = Net(input)
                    target = target.reshape([net_output.size()[0], net_output.size()[1]])
                '''

                if cfg.NET_MODEL.TYPE == 'SFCN':
                    net_output = Net(input)[0]
                    target_age = target_age.reshape([net_output.size()[0], net_output.size()[1], 1, 1, 1])
                else:
                    net_output = Net(input)
                    #target_age = target_age.reshape([net_output.size()[0]])
                # model = MSD3D(cfg)
                #net_output = Net(input)
                #target = target.reshape([net_output.size()[0], net_output.size()[1]])
                #exec(exec_line)
                # print(target.size())
                loss = criterion(net_output, target_age)
                #print(net_output[0], target_age[0])
                age_x, age_difference = age_diff_mse(net_output, target_age)
                epoch_loss += loss.item()
                epoch_age_loss += age_difference.item()
                # Scales the loss, and calls backward()
                # to create scaled gradients
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if (iteration % 20 == 0):
                    logger.info("WHOLE ===> Epoch[{}]({}/{}):Training Batch Loss: {:.8f}".format(epoch, iteration,
                                                                                                len(training_data_loader),
                                                                                                loss.item()))
            logger.info("WHOLE ===> Epoch {} Complete: "
                        "WHOLE Avg. Training Loss: {:.8f}".format(epoch, epoch_loss / len(training_data_loader)))
            logger.info("WHOLE  ===> Epoch {} Complete: "
                        "WHOLE Avg. Training Age Loss: {:.8f}".format(epoch, epoch_age_loss / len(training_data_loader)))

            time_elapsed = time.time() - time_start
            logger.info('WHOLE Time elapse  {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            # predict the test set
            logger.info('===> start validation')
            torch.cuda.empty_cache()
            test_epoch_loss = 0
            age_diff_loss = 0

            # write age prediction
            predict_file = cfg.NET_TRAIN.WHOLE_NET_OUTPUT_DIR + f'{epoch}_' + time.strftime("%Y-%m-%d") + '_age_predict.csv'
            predict_file = open(predict_file, "w")  # 创建csv文件
            writer = csv.writer(predict_file)
            writer.writerow(['user id', 'predict age', 'real age', 'age difference'])
            with torch.no_grad():
                for iteration, batch in enumerate(validate_data_loader, 1):
                    input = batch[5].to(device)
                    target = batch[2].to(device)
                    userid = batch[3]
                    target_age = batch[4].to(device)

                    net_output = Net(input)

                    test_loss = criterion(net_output, target_age)
                    age_x, age_difference = age_diff_mse(net_output, target_age)
                    test_epoch_loss += test_loss.item()
                    age_diff_loss += age_difference.item()
                    writer.writerow([userid, age_x.item(), target_age.item(), age_difference.item()])
                    # print("===> Epoch[{}]({}/{}):Validation Batch Loss: {:.8f}".format(epoch, iteration, len(testing_data_loader),  test_loss.item()))
            avg_test_loss = test_epoch_loss / len(validate_data_loader)
            ave_age_loss = age_diff_loss / len(validate_data_loader)
            logger.info("WHOLE ===> Epoch {} Complete: Avg. Validation Loss: {:.8f}".format(epoch, avg_test_loss))
            logger.info("WHOLE ===> Epoch {} Complete: Avg. Validation age difference: {:.8f}".format(epoch, ave_age_loss))

            predict_file.close()

            loss_all = epoch_loss / len(training_data_loader)
            if cfg.NET_TRAIN.VISDOM:
                vis.plot('WHOLE Average Batch loss', loss_all)
                record_log = {'epoch': epoch, 'Avg training batch loss': epoch_loss / len(training_data_loader)}
                vis.log(record_log)
            if cfg.NET_TRAIN.USE_GPU:
                torch.cuda.synchronize()
            time_elapsed = time.time() - time_start
            logger.info('WHOLE Training elapse  {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            logger.info(str(epoch) + ', ' + str(epoch_loss / len(training_data_loader)) + '\n')

            # if (epoch % 10 == 0):  # if (epoch % 10 == 0):
            #    torch.save(Net.state_dict(), cfg.NET_OUTPUT_DIR + 'pytorch_model_' + str(epoch)+time.strftime("%Y-%m-%d"))
            #    pass
            # decide how to save
            if ave_age_loss <= validate_last_loss:
                torch.save(Net.state_dict(),
                           cfg.NET_TRAIN.WHOLE_NET_OUTPUT_DIR + 'pytorch_model_' + str(epoch) + '_' + time.strftime("%Y-%m-%d"))
                model_save_path = cfg.NET_TRAIN.WHOLE_NET_OUTPUT_DIR + 'pytorch_model_' + str(epoch) + '_' + time.strftime("%Y-%m-%d")

                validate_last_loss = ave_age_loss

            if cfg.NET_TRAIN.EARLY_STOPPING:
                early_stopping(avg_test_loss, Net)

                if early_stopping.early_stop:
                    logger.info(f"WHOLE: Early stopped after {epoch} epoch")
                    break
        trained_model_dict['whole'] = model_save_path
        #return trained_model_dict
        Net.load_state_dict(torch.load(model_save_path))
        logger.info('===> start predicting training data')
        training_data_loader = DataLoader(dataset=train_set, num_workers=cfg.NET_DATALOADER.NUM_WORKERS,
                                          batch_size=1, shuffle=False,
                                          pin_memory=cfg.NET_DATALOADER.PIN_MEMORY,
                                          drop_last=cfg.NET_DATALOADER.DROP_LAST)
        torch.cuda.empty_cache()
        training_epoch_loss = 0
        age_diff_loss = 0

        # write age prediction
        predict_file = cfg.NET_TRAIN.WHOLE_NET_OUTPUT_DIR + f'{epoch}_' + time.strftime("%Y-%m-%d") + '_trainingset_age_predict.csv'
        predict_file = open(predict_file, "w")  # 创建csv文件
        writer = csv.writer(predict_file)
        writer.writerow(['user id', 'predict age', 'real age', 'age difference'])
        with torch.no_grad():
            for iteration, batch in enumerate(training_data_loader, 1):
                input = batch[5].to(device)
                target = batch[2].to(device)
                userid = batch[3]
                target_age = batch[4].to(device)

                net_output = Net(input)

                training_loss = criterion(net_output, target_age)
                age_x, age_difference = age_diff_mse(net_output, target_age)
                training_epoch_loss += training_loss.item()
                age_diff_loss += age_difference.item()
                writer.writerow([userid, age_x.item(), target_age.item(), age_difference.item()])
                # print("===> Epoch[{}]({}/{}):Validation Batch Loss: {:.8f}".format(epoch, iteration, len(testing_data_loader),  test_loss.item()))
        avg_training_loss = training_epoch_loss / len(training_data_loader)
        ave_age_loss = age_diff_loss / len(training_data_loader)
        logger.info("WHOLE ===> Epoch {} Complete: Avg. Training Loss: {:.8f}".format(epoch, avg_training_loss))
        logger.info("WHOLE ===> Epoch {} Complete: Avg. Training age difference: {:.8f}".format(epoch, ave_age_loss))

        predict_file.close()

    elif cfg.NET_TRAIN.TRAIN_TYPE == 'left_right':
        # --------------------------------------------------------------------------------------------------
        logger.info('===> Building left model')

        if cfg.NET_MODEL.TYPE == 'SFCN':
            model = sfcn_leftright.SFCN(output_dim=cfg.NET_MODEL.CLASS_NUM)
            #exec_line = 'net_output = Net(input)[0]\ntarget = target.reshape([net_output.size()[0], net_output.size()[1], 1, 1, 1])'
        elif cfg.NET_MODEL.TYPE == 'RESNET3D18':
            model = resnet3d_leftright.resnet18(num_classes=cfg.NET_MODEL.CLASS_NUM)
            #exec_line = 'net_output = Net(input)\ntarget = target.reshape([net_output.size()[0], net_output.size()[1]])'
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
        # set optimizer
        optimizer = makeSolver(cfg, Net)

        # Waiting for the end of all the center program
        if cfg.NET_TRAIN.USE_GPU:
            torch.cuda.synchronize()

        time_start = time.time()

        # visualization
        if cfg.NET_TRAIN.VISDOM:
            vis = Visualizer(env=cfg.VISUALIZER.ENV)

        # Auto Mixed Precision Training
        scaler = GradScaler()

        logger.info('===> start training')

        # init learning rate
        lr = cfg.SOLVER.INIT_LR

        # early stopping
        if cfg.NET_TRAIN.EARLY_STOPPING is not None:
            early_stopping = EarlyStopping(patience=cfg.NET_TRAIN.PATIENCE, verbose=True)

        age_range = torch.Tensor(cfg.NET_DATALOADER.BIN_RANGE)

        bc = np.arange(cfg.NET_DATALOADER.BIN_RANGE[0], cfg.NET_DATALOADER.BIN_RANGE[1]) + 0.5
        bc = torch.tensor(bc, dtype=torch.float32).to(device)

        # start training if trainig is interrupted
        epoch_append = 0
        if cfg.NET_TRAIN.FINE_TUNE:
            Net.load_state_dict(torch.load(cfg.NET_TRAIN.LEFT_PRETRAIN_MODEL_PATH))
            epoch_append = int(cfg.NET_TRAIN.LEFT_PRETRAIN_MODEL_PATH.split('_')[3])

        validate_last_loss = 100000

        for epoch in range(1 + epoch_append, cfg.NET_TRAIN.EPOCHS + 1 + epoch_append):
            epoch_loss = 0
            epoch_age_loss = 0
            if epoch > 0 and epoch % cfg.SOLVER.LR_PERIOD == 0 and lr > 1e-8:  # 每lr_period个epoch，学习率衰减一次
                lr = lr * cfg.SOLVER.LR_DECAY
                for param_group in optimizer.param_groups[6:]:
                    param_group['lr'] = lr
            for iteration, batch in enumerate(training_data_loader, 1):
                input = batch[0].to(device)
                target = batch[2].to(device)
                userid = batch[3]
                target_age = batch[4].to(device)
                #print('target_age', target_age)
                optimizer.zero_grad()
                # Casts operations to mixed precision
                # with torch.cuda.amp.autocast():
                #net_output = Net(input)
                #print(input.size())
                #print(net_output.size())
                #print(net_output)
                #target = target.reshape([net_output.size()[0], net_output.size()[1]])
                #exec(exec_line)
                if cfg.NET_MODEL.TYPE == 'SFCN':
                    net_output = Net(input)[0]
                    target = target.reshape([net_output.size()[0], net_output.size()[1], 1, 1, 1])
                else:
                    net_output = Net(input)
                    target = target.reshape([net_output.size()[0], net_output.size()[1]])
                #print(target.size())
                loss = criterion(net_output, target)
                age_x, age_difference = age_diff(net_output, target_age, bc)
                epoch_loss += loss.item()
                epoch_age_loss += age_difference.item()
                # Scales the loss, and calls backward()
                # to create scaled gradients
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if (iteration % 20 == 0):
                    logger.info("LEFT ===> Epoch[{}]({}/{}):Training Batch Loss: {:.8f}".format(epoch, iteration,
                                                                                     len(training_data_loader),
                                                                                     loss.item()))
            logger.info("LEFT ===> Epoch {} Complete: "
                        "LEFT Avg. Training Loss: {:.8f}".format(epoch, epoch_loss / len(training_data_loader)))
            logger.info("LEFT  ===> Epoch {} Complete: "
                        "LEFT Avg. Training Age Loss: {:.8f}".format(epoch, epoch_age_loss / len(training_data_loader)))

            time_elapsed = time.time() - time_start
            logger.info('LEFT Time elapse  {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            # predict the test set
            logger.info('===> start validation')
            torch.cuda.empty_cache()
            test_epoch_loss = 0
            age_diff_loss = 0

            # write age prediction
            predict_file = cfg.NET_TRAIN.LEFT_NET_OUTPUT_DIR + f'{epoch}_'+time.strftime("%Y-%m-%d")+'_age_predict.csv'
            predict_file = open(predict_file, "w")  # 创建csv文件
            writer = csv.writer(predict_file)
            writer.writerow(['user id', 'predict age', 'real age', 'age difference'])
            with torch.no_grad():
                for iteration, batch in enumerate(validate_data_loader, 1):
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
            avg_test_loss = test_epoch_loss / len(validate_data_loader)
            ave_age_loss = age_diff_loss / len(validate_data_loader)
            logger.info("LEFT ===> Epoch {} Complete: Avg. Validation Loss: {:.8f}".format(epoch, avg_test_loss))
            logger.info("LEFT ===> Epoch {} Complete: Avg. Validation age difference: {:.8f}".format(epoch, ave_age_loss))

            predict_file.close()

            loss_all = epoch_loss / len(training_data_loader)
            if cfg.NET_TRAIN.VISDOM:
                vis.plot('LEFT Average Batch loss', loss_all)
                record_log = {'epoch':epoch, 'Avg training batch loss':epoch_loss/len(training_data_loader)}
                vis.log(record_log)
            if cfg.NET_TRAIN.USE_GPU:
                torch.cuda.synchronize()
            time_elapsed = time.time() - time_start
            logger.info('LEFT Training elapse  {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            logger.info(str(epoch) + ', ' + str(epoch_loss / len(training_data_loader)) + '\n')

            #if (epoch % 10 == 0):  # if (epoch % 10 == 0):
            #    torch.save(Net.state_dict(), cfg.NET_OUTPUT_DIR + 'pytorch_model_' + str(epoch)+time.strftime("%Y-%m-%d"))
            #    pass
            # decide how to save
            if ave_age_loss <= validate_last_loss:
                torch.save(Net.state_dict(),
                           cfg.NET_TRAIN.LEFT_NET_OUTPUT_DIR + 'pytorch_model_' + str(epoch) + '_' + time.strftime("%Y-%m-%d"))
                model_save_path = cfg.NET_TRAIN.LEFT_NET_OUTPUT_DIR + 'pytorch_model_' + str(epoch) + '_' + time.strftime("%Y-%m-%d")

                validate_last_loss = ave_age_loss

            if cfg.NET_TRAIN.EARLY_STOPPING:
                early_stopping(avg_test_loss, Net)

                if early_stopping.early_stop:
                    logger.info(f"LEFT: Early stopped after {epoch} epoch")
                    break
        trained_model_dict['left'] = model_save_path

        # --------------------------------------------------------------------------------------------------
        logger.info('===> Building right model')

        if cfg.NET_MODEL.TYPE == 'SFCN':
            model = sfcn_leftright.SFCN(output_dim=cfg.NET_MODEL.CLASS_NUM)
            #exec_line = 'net_output = Net(input)[0]\ntarget = target.reshape([net_output.size()[0], net_output.size()[1], 1, 1, 1])'
        elif cfg.NET_MODEL.TYPE == 'RESNET3D18':
            model = resnet3d_leftright.resnet18(num_classes=cfg.NET_MODEL.CLASS_NUM)
            #exec_line = 'net_output = Net(input)\ntarget = target.reshape([net_output.size()[0], net_output.size()[1]])'
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
        # set optimizer
        optimizer = makeSolver(cfg, Net)

        # Waiting for the end of all the center program
        if cfg.NET_TRAIN.USE_GPU:
            torch.cuda.synchronize()

        time_start = time.time()

        # visualization
        if cfg.NET_TRAIN.VISDOM:
            vis = Visualizer(env=cfg.VISUALIZER.ENV)

        # Auto Mixed Precision Training
        scaler = GradScaler()

        logger.info('===> start training')

        # init learning rate
        lr = cfg.SOLVER.INIT_LR

        # early stopping
        if cfg.NET_TRAIN.EARLY_STOPPING is not None:
            early_stopping = EarlyStopping(patience=cfg.NET_TRAIN.PATIENCE, verbose=True)

        age_range = torch.Tensor(cfg.NET_DATALOADER.BIN_RANGE)

        bc = np.arange(cfg.NET_DATALOADER.BIN_RANGE[0], cfg.NET_DATALOADER.BIN_RANGE[1]) + 0.5
        bc = torch.tensor(bc, dtype=torch.float32).to(device)

        # start training if trainig is interrupted
        epoch_append = 0
        if cfg.NET_TRAIN.FINE_TUNE:
            Net.load_state_dict(torch.load(cfg.NET_TRAIN.RIGHT_PRETRAIN_MODEL_PATH))
            epoch_append = int(cfg.NET_TRAIN.RIGHT_PRETRAIN_MODEL_PATH.split('_')[3])

        validate_last_loss = 100000

        for epoch in range(1 + epoch_append, cfg.NET_TRAIN.EPOCHS + 1 + epoch_append):
            epoch_loss = 0
            epoch_age_loss = 0
            if epoch > 0 and epoch % cfg.SOLVER.LR_PERIOD == 0 and lr > 1e-8:  # 每lr_period个epoch，学习率衰减一次
                lr = lr * cfg.SOLVER.LR_DECAY
                for param_group in optimizer.param_groups[6:]:
                    param_group['lr'] = lr
            for iteration, batch in enumerate(training_data_loader, 1):
                input = batch[1].to(device)
                target = batch[2].to(device)
                userid = batch[3]
                target_age = batch[4].to(device)
                # print('target_age', target_age)
                optimizer.zero_grad()
                # Casts operations to mixed precision
                # with torch.cuda.amp.autocast():
                #net_output = Net(input)
                #target = target.reshape([net_output.size()[0], net_output.size()[1]])
                if cfg.NET_MODEL.TYPE == 'SFCN':
                    net_output = Net(input)[0]
                    target = target.reshape([net_output.size()[0], net_output.size()[1], 1, 1, 1])
                else:
                    net_output = Net(input)
                    target = target.reshape([net_output.size()[0], net_output.size()[1]])
                # print(target.size())
                loss = criterion(net_output, target)
                age_x, age_difference = age_diff(net_output, target_age, bc)
                epoch_loss += loss.item()
                epoch_age_loss += age_difference.item()
                # Scales the loss, and calls backward()
                # to create scaled gradients
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if (iteration % 20 == 0):
                    logger.info("RIGHT ===> Epoch[{}]({}/{}):Training Batch Loss: {:.8f}".format(epoch, iteration,
                                                                                                len(training_data_loader),
                                                                                                loss.item()))
            logger.info("RIGHT ===> Epoch {} Complete: "
                        "RIGHT Avg. Training Loss: {:.8f}".format(epoch, epoch_loss / len(training_data_loader)))
            logger.info("RIGHT  ===> Epoch {} Complete: "
                        "RIGHT Avg. Training Age Loss: {:.8f}".format(epoch, epoch_age_loss / len(training_data_loader)))

            time_elapsed = time.time() - time_start
            logger.info('RIGHT Time elapse  {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            # predict the test set
            logger.info('===> start validation')
            torch.cuda.empty_cache()
            test_epoch_loss = 0
            age_diff_loss = 0

            # write age prediction
            predict_file = cfg.NET_TRAIN.RIGHT_NET_OUTPUT_DIR + f'{epoch}_' + time.strftime("%Y-%m-%d") + 'age_predict.csv'
            predict_file = open(predict_file, "w")  # 创建csv文件
            writer = csv.writer(predict_file)
            writer.writerow(['user id', 'predict age', 'real age', 'age difference'])
            with torch.no_grad():
                for iteration, batch in enumerate(validate_data_loader, 1):
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
            avg_test_loss = test_epoch_loss / len(validate_data_loader)
            ave_age_loss = age_diff_loss / len(validate_data_loader)
            logger.info("RIGHT ===> Epoch {} Complete: Avg. Validation Loss: {:.8f}".format(epoch, avg_test_loss))
            logger.info(
                "RIGHT ===> Epoch {} Complete: Avg. Validation age difference: {:.8f}".format(epoch, ave_age_loss))

            predict_file.close()

            loss_all = epoch_loss / len(training_data_loader)
            if cfg.NET_TRAIN.VISDOM:
                vis.plot('RIGHT Average Batch loss', loss_all)
                record_log = {'epoch': epoch, 'Avg training batch loss': epoch_loss / len(training_data_loader)}
                vis.log(record_log)
            if cfg.NET_TRAIN.USE_GPU:
                torch.cuda.synchronize()
            time_elapsed = time.time() - time_start
            logger.info('RIGHT Training elapse  {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            logger.info(str(epoch) + ', ' + str(epoch_loss / len(training_data_loader)) + '\n')

            # if (epoch % 10 == 0):  # if (epoch % 10 == 0):
            #    torch.save(Net.state_dict(), cfg.NET_OUTPUT_DIR + 'pytorch_model_' + str(epoch)+time.strftime("%Y-%m-%d"))
            #    pass
            # decide how to save
            if ave_age_loss <= validate_last_loss:
                torch.save(Net.state_dict(),
                           cfg.NET_TRAIN.RIGHT_NET_OUTPUT_DIR + 'pytorch_model_' + str(epoch) + '_' + time.strftime("%Y-%m-%d"))
                model_save_path = cfg.NET_TRAIN.RIGHT_NET_OUTPUT_DIR + 'pytorch_model_' + str(epoch) + '_' + time.strftime("%Y-%m-%d")

                validate_last_loss = ave_age_loss #

            if cfg.NET_TRAIN.EARLY_STOPPING:
                early_stopping(avg_test_loss, Net)

                if early_stopping.early_stop:
                    logger.info(f"RIGHT: Early stopped after {epoch} epoch")
                    break

        trained_model_dict['right'] = model_save_path

    return trained_model_dict



















