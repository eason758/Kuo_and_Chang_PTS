import torch
import torch.nn as nn
import torch.utils.data as Data
import json
import sys
import logging
import os
import matplotlib.pyplot as plt
import datetime
import glob
import numpy as np
from shutil import copyfile
from testing import dump_testing_result, get_performance_by_predicted_action, agg_all_cluster_result
from dl_utlis import data_preprocess
import pandas as pd
import configparser

from utils import get_optimal_actions_in_specific_pair  # noqa: E402


EPOCH = 350
BATCH_SIZE = 1024
MODEL_FOLDER_ROOT = r'C:\Users\eason\Desktop\PTDQN-main\allen-kuo-DL'


def DataTransfer(train_data, train_label, test_data=None, test_label=None):
    train_data = torch.FloatTensor(train_data).cuda()
    train_label = torch.LongTensor(train_label).cuda()
    torch_dataset_train = Data.TensorDataset(train_data, train_label)
    loader_train = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    if test_data is not None and test_label is not None:
        test_data = torch.FloatTensor(test_data).cuda()
        test_label = torch.LongTensor(test_label).cuda()
        torch_dataset_test = Data.TensorDataset(test_data, test_label)
        loader_test = Data.DataLoader(
            dataset=torch_dataset_test,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    else:
        loader_test = None
    return loader_train, loader_test


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)


class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        # out += residual
        return out1


class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        # out += residual
        return out1


class MSResNet(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=25):
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64
        super(MSResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 128, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)
        # maxplooing kernel size: 16, 11, 6
        self.maxpool3 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)
        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 256, layers[2], stride=2)
        # self.layer5x5_4 = self._make_layer5(BasicBlock5x5, 512, layers[3], stride=2)
        self.maxpool5 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)
        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 128, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 256, layers[2], stride=2)
        # self.layer7x7_4 = self._make_layer7(BasicBlock7x7, 512, layers[3], stride=2)
        self.maxpool7 = nn.AvgPool1d(kernel_size=6, stride=1, padding=0)
        # self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(256*3, num_classes)
        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))
        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))
        return nn.Sequential(*layers)

    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))
        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        x = self.maxpool3(x)
        y = self.layer5x5_1(x0)
        y = self.layer5x5_2(y)
        y = self.layer5x5_3(y)
        y = self.maxpool5(y)
        z = self.layer7x7_1(x0)
        z = self.layer7x7_2(z)
        z = self.layer7x7_3(z)
        z = self.maxpool7(z)
        out = torch.cat([x, y, z], dim=1)
        out = out.squeeze()
        out1 = self.fc(out)
        return out1


def model_train(folder_name, loader_train, loader_test, validate_action_data, given_action, second_stage=False,
                pre_trained_model_folder=None):
    logging.info(f'model_train given_action: {given_action}')
    if second_stage:
        num_classes = 2
        sub_folder = 'second_stage'
    else:
        num_classes = len(given_action)
        sub_folder = 'first_stage'
    logging.info(f'model_train num_classes: {num_classes}')
    # multi-scale ResNet paper: "Multiscale Kernel Based Residual Convolutional Neural Network for Motor Fault Diagnosis
    #                            Under Nonstationary Conditions" doi: 10.1109/TII.2019.2941868.
    # multi-scale ResNet source code: https://github.com/geekfeiw/Multi-Scale-1D-ResNet
    # Adopt the original code hyper-parameter
    if pre_trained_model_folder is None:
        msresnet = MSResNet(input_channel=3, layers=[1, 1, 1, 1], num_classes=num_classes)
    else:
        if os.path.exists(f"{MODEL_FOLDER_ROOT}/{pre_trained_model_folder}/first_stage"):
            tmp = f'{pre_trained_model_folder}/first_stage'
        else:
            tmp = pre_trained_model_folder
        if os.path.isfile(f"{MODEL_FOLDER_ROOT}/{tmp}/best_win_rate.pkl"):
            logging.info(f'Load {MODEL_FOLDER_ROOT}/{tmp}/best_win_rate.pkl!')
            msresnet = torch.load(f"{MODEL_FOLDER_ROOT}/{tmp}/best_win_rate.pkl")
        else:
            logging.info(f'{MODEL_FOLDER_ROOT}/{tmp}/best_win_rate.pkl does not exist! '
                         f'Use final.pkl instead!')
            msresnet = torch.load(f"{MODEL_FOLDER_ROOT}/{tmp}/final.pkl")
    msresnet = msresnet.cuda()
    criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    optimizer = torch.optim.Adam(msresnet.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300], gamma=0.1)
    train_loss = []
    train_acc = []
    validate_acc = []
    best_win_rate = -1
    validate_win_rate_list = list()
    for epoch in range(EPOCH):
        total_train = 0
        total_validate = 0
        correct_train = 0
        correct_validate = 0
        action_list = list()
        msresnet.train()
        scheduler.step()
        for step, (batch_x, batch_y) in enumerate(loader_train):
            output = msresnet(batch_x)
            #print(output.shape, batch_y.shape)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output, 1)
            total_train += batch_y.nelement()
            correct_train += predicted.eq(batch_y).sum().item()
        train_accuracy = 100 * correct_train / total_train
        train_loss.append(loss.item())
        train_acc.append(train_accuracy)

        logging.info(f'Epoch {epoch}, train Loss: {loss.item():.5f} Training Accuracy: {train_accuracy}')
        if not second_stage:
            # Only the first stage need to record the win rate with validation data.
            try:
                for step, (batch_x, batch_y) in enumerate(loader_test):
                    output = msresnet(batch_x).cuda()
                    loss = criterion(output, batch_y).cuda()
                    _, predicted = torch.max(output, 1)
                    total_validate += batch_y.nelement()
                    correct_validate += predicted.eq(batch_y).sum().item()
                    action_choose = predicted.cpu().numpy()
                    action_choose = action_choose.tolist()
                    action_list.append(action_choose)
                action_list = sum(action_list, [])
                if len(action_list) > 0 and len(validate_action_data) > 0:
                    validate_accuracy = 100 * correct_validate / total_validate
                    validate_acc.append(validate_accuracy)
                    win_count = 0
                    total_open_number = 0
                    logging.info(f'validate label len:{len(action_list)} validate data len:{len(validate_action_data)}')
                    if len(action_list) != len(validate_action_data):
                        raise ValueError('Label len does not match data len!')
                    for action_idx, data in zip(action_list, validate_action_data):
                        target_action = given_action[action_idx]
                        if data[f"{target_action[0]},{target_action[1]}"]['record'] != 0:
                            total_open_number += 1
                        if data[f"{target_action[0]},{target_action[1]}"]['half_tax_profit'] > 0:
                            win_count += 1
                    if total_open_number != 0:
                        win_rate = win_count / total_open_number
                    else:
                        win_rate = -2
                    validate_win_rate_list.append(win_rate)
                    if win_rate >= best_win_rate and train_accuracy >= 95:
                        logging.info(f'EPOCH:{epoch} original win_rate:{best_win_rate} new win_rate:{win_rate}')
                        torch.save(msresnet, f"{MODEL_FOLDER_ROOT}/{folder_name}/{sub_folder}/best_win_rate.pkl")
                        best_win_rate = win_rate
                    print(f'{epoch}/{EPOCH}, train_loss:{loss.item()} , train_acc:{train_accuracy}, val_acc:{validate_acc}')
            except Exception as e:
                logging.info(f'error: {e}')
                pass
        if epoch == EPOCH - 1:
            torch.save(msresnet, f"{MODEL_FOLDER_ROOT}/{folder_name}/{sub_folder}/final.pkl")
    draw(f'{folder_name}/{sub_folder}', train_loss, train_acc, validate_acc, validate_win_rate_list)


def draw(folder, train_loss, train_acc, validate_acc, validate_win_rate_list):
    plt.figure(figsize=(10, 6))
    plt.xlabel('Epoch')
    plt.ylabel('Train_loss')
    plt.plot(train_loss)
    plt.tight_layout()
    plt.savefig(f'{MODEL_FOLDER_ROOT}/{folder}/train_loss.jpg')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.xlabel('Epoch')
    plt.ylabel('Train_Acc(%)')
    plt.plot(train_acc)
    plt.tight_layout()
    plt.savefig(f'{MODEL_FOLDER_ROOT}/{folder}/train_acc.jpg')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.xlabel('Epoch')
    plt.ylabel('validate_Acc(%)')
    plt.plot(validate_acc)
    plt.tight_layout()
    plt.savefig(f'{MODEL_FOLDER_ROOT}/{folder}/validate_acc.jpg')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.xlabel('Epoch')
    plt.ylabel('validate_win_rate')
    plt.plot(validate_win_rate_list)
    plt.tight_layout()
    plt.savefig(f'{MODEL_FOLDER_ROOT}/{folder}/validate_win_rate.jpg')
    plt.close()


def get_trained_model_folder(top_n):
    if top_n <= 29:
        source_trained_model_folder = '2022/20220425-P149C2/2022-04-15-13-53-00-2017_20029_29_150'
        pre_trained_model_folder = '20220704-P66/tmp1/2022-06-30-13-51-55-2017_200_29_29'
    elif top_n <= 60:
        source_trained_model_folder = '2022/20220425-P149C2/2022-04-15-14-02-11-2017_20060_60_150'
        pre_trained_model_folder = '20220704-P66/tmp1/2022-06-30-13-55-29-2017_200_60_60'
    elif top_n <= 131:
        source_trained_model_folder = '2022/20220425-P149C2/2022-04-15-14-18-18-2017_200131_131_150'
        pre_trained_model_folder = '20220704-P66/tmp1/2022-06-30-14-00-48-2017_200_131_131'
    elif top_n <= 229:
        source_trained_model_folder = '2022/20220425-P149C2/2022-04-15-14-49-38-2017_200229_229_150'
        pre_trained_model_folder = '20220704-P66/tmp1/2022-06-30-14-09-55-2017_200_229_229'
    elif top_n <= 391:
        source_trained_model_folder = '2022/20220425-P149C2/2022-04-15-15-40-43-2017_200391_391_150'
        pre_trained_model_folder = '20220704-P66/tmp1/2022-06-30-14-23-46-2017_200_391_391'
    elif top_n <= 655:
        source_trained_model_folder = '2022/20220425-P149C2/2022-04-15-13-52-59-2017_200655_655_150'
        pre_trained_model_folder = '20220704-P66/tmp1/2022-06-30-14-44-33-2017_200_655_655'
    elif top_n <= 1076:
        source_trained_model_folder = '2022/20220425-P149C2/2022-04-15-15-52-29-2017_2001076_1076_150'
        pre_trained_model_folder = '20220704-P66/tmp1/2022-06-30-15-15-29-2017_200_1076_1076'
    elif top_n <= 1661:
        source_trained_model_folder = '2022/20220425-P149C2/2022-04-15-18-46-56-2017_2001661_1661_150'
        pre_trained_model_folder = '20220704-P66/tmp1/2022-06-30-16-01-31-2017_200_1661_1661'
    elif top_n <= 2394:
        source_trained_model_folder = '2022/20220425-P149C2/2022-04-15-13-53-04-2017_2002394_2394_150'
        pre_trained_model_folder = '20220704-P66/tmp1/2022-06-30-17-03-44-2017_200_2394_2394'
    elif top_n <= 3354:
        source_trained_model_folder = '2022/20220425-P149C2/2022-04-15-19-04-26-2017_2003354_3354_150'
        pre_trained_model_folder = '20220704-P66/tmp1/2022-06-30-20-35-46-2017_200_3354_3354'
    elif top_n <= 4627:
        source_trained_model_folder = '2022/20220425-P149C2/2022-04-15-13-52-57-2017_2004627_4627_150'
        pre_trained_model_folder = '20220704-P66/tmp1/2022-06-30-22-15-32-2017_200_4627_4627'
    else:
        raise ValueError(f'Unknown top_n:{top_n}!')
    return source_trained_model_folder, pre_trained_model_folder


def main(test_year, test_start_month=1, n_test_month=24,
             validate_start_year=2017, second_stage_months=12, second_stage_start_month=1,
             validate_start_month=11, validate_months=2,
             training_months=22,
         top_n=None, two_stage=False, load_pre_trained_model=False):
    """ Refactor the two-stage model proposed by Allen Kuo (2020).

    The original source code: https://github.com/allenfishkuo/PT_Final
    In this version, we generate the unique folder to keep the training information.

    Args:
        test_year (int): The start of the testing year.
        top_n (None or int): If not `None`, load the Top N frequent pairs. Otherwise would use load all pairs.
        two_stage (bool): If `False`, only train the first stage.
        load_pre_trained_model (bool): If `True`, load pre-trained model to fine-tune
    Returns:
        str: the unique folder name
    """
    test_info = dict()
    now = datetime.datetime.now()
    date = now.strftime('%Y-%m-%d-%H-%M-%S')
    specific_model_type = 0
    model_type_json_file = None
    if top_n is None:
        stock1 = None
        stock2 = None
    else:
        with open("../stock_info/HsinHuaChang_with_trend_TOP_N_STOCK1_IN_2015_2016.txt") as F:
            top_stock1 = [line.strip() for line in F]
        with open("../stock_info/HsinHuaChang_with_trend_TOP_N_STOCK2_IN_2015_2016.txt") as F:
            top_stock2 = [line.strip() for line in F]
        stock1 = top_stock1[top_n - 1]
        stock2 = top_stock2[top_n - 1]


    stock_price_data_folder = [r'Data\HsinHuaChang_reverse_random_300_new_std_price_pickle',]
    #                   'HsinHuaChang_reverse_random_300_trend_stationary_new_std_pickle']

    data_folder = ['Data\HsinHuaChang_reverse_random_300_new_std_pickle',]
    #                   'HsinHuaChang_reverse_random_300_trend_stationary_new_std_pickle']

    # Create unique folder to record the training setting
    folder_name = f'tmp7/{date}-{test_year}_{EPOCH}'
    if specific_model_type:
        folder_name += f'model_type={specific_model_type}'
    folder_name += f'_{stock1}_{stock2}'
    if not os.path.exists(f'{MODEL_FOLDER_ROOT}/{folder_name}'):
        os.makedirs(f'{MODEL_FOLDER_ROOT}/{folder_name}')
        os.makedirs(f'{MODEL_FOLDER_ROOT}/{folder_name}/first_stage')
        os.makedirs(f'{MODEL_FOLDER_ROOT}/{folder_name}/second_stage')

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        filename=f'{MODEL_FOLDER_ROOT}/{folder_name}/log.txt', level=logging.INFO)
    program_file = f'{MODEL_FOLDER_ROOT}/{folder_name}/program/'
    if not os.path.exists(program_file):
        os.makedirs(program_file)
        for file in sorted(glob.glob('*.py')):
            copyfile(file, program_file + os.path.basename(file))


    if top_n is not None:
        # Use trained model folder to get target actions
        source_trained_model_folder, pre_trained_model_folder = get_trained_model_folder(top_n)
        if not load_pre_trained_model:
            pre_trained_model_folder = None
        with open(f'D:/ptdqn_model/{source_trained_model_folder}/agent_info.json', 'r') as f:
            PRE_TRAINED_MODEL_DATA = json.load(f)
            given_action = PRE_TRAINED_MODEL_DATA['GIVEN_ACTION']
            given_action.sort(reverse=True)
            logging.info(f'Use {source_trained_model_folder} get given action: {given_action}')
    else:
        import pickle
        if os.path.exists(r'C:\Users\eason\Desktop\PTDQN-main\allen-kuo-DL\actions.pkl'):
            with open(r'C:\Users\eason\Desktop\PTDQN-main\allen-kuo-DL\actions.pkl', 'rb') as f:
                given_action = pickle.load(f)
        else:
            # Use representation labeling mechanism `HighFreq` to get the target actions
            _, given_action = get_optimal_actions_in_specific_pair(validate_start_year, validate_start_month,
                                                                   training_months, data_folder)
            given_action.sort(reverse=True)
            # Non-opening action need to cast to int to match the key in data folder
            try:
                given_action[given_action.index([1000000, 2500000])] = \
                    [int(x) for x in given_action[given_action.index([1000000, 2500000])]]

                with open(r'C:\Users\eason\Desktop\PTDQN-main\allen-kuo-DL\actions.pkl', 'wb') as f:
                    pickle.dump(given_action, f)
            except ValueError:
                print('Non-opening action does not in given action!')
        logging.info(f'Use training data pairs to get given action: {given_action}')
        pre_trained_model_folder = None

    test_info['test_year'] = test_year
    test_info['test_month'] = test_start_month
    test_info['test_months'] = n_test_month
    test_info['given_action'] = given_action
    test_info['stock1'] = stock1
    test_info['stock2'] = stock2
    test_info['specific_model_type'] = specific_model_type
    test_info['model_type_json_file'] = model_type_json_file
    test_info['stock_price_data_folder'] = stock_price_data_folder
    test_info['data_folder'] = data_folder
    test_info['two_stage'] = two_stage
    test_info['pre_trained_model_folder'] = pre_trained_model_folder

    with open(f'{MODEL_FOLDER_ROOT}/{folder_name}/test_info.json', 'w') as file:
        json.dump(test_info, file)

    logging.info('Load first stage training and validation data set!')

    training_data, training_label, validate_data, validate_label, validate_action_data = \
        data_preprocess(stock_price_data_folder, data_folder, validate_start_year, validate_start_month,
                        validate_months, training_months, given_action, stock1, stock2,
                        specific_model_type, model_type_json_file)


    loader_train, loader_test = DataTransfer(training_data, training_label, validate_data, validate_label)

    model_train(folder_name, loader_train, loader_test, validate_action_data, given_action,
                pre_trained_model_folder=pre_trained_model_folder)

    if two_stage:
        logging.info('Load second stage training data set!')
        second_training_data, second_training_label, _, _, second_training_action_data = \
            data_preprocess(stock_price_data_folder, data_folder, test_year, second_stage_start_month, 0, second_stage_months,
                            given_action, stock1, stock2, specific_model_type, model_type_json_file, second_stage=True)

        if os.path.isfile(f"{MODEL_FOLDER_ROOT}/{folder_name}/first_stage/best_win_rate.pkl"):
            trained_model = torch.load(f"{MODEL_FOLDER_ROOT}/{folder_name}/first_stage/best_win_rate.pkl")
        else:
            logging.info(f'{MODEL_FOLDER_ROOT}/{folder_name}/first_stage/best_win_rate.pkl does not exist! '
                         f'Use final.pkl instead!')
            trained_model = torch.load(f"{MODEL_FOLDER_ROOT}/{folder_name}/first_stage/final.pkl")

        final_result = get_performance_by_predicted_action(trained_model, second_training_data, second_training_label,
                                                           second_training_action_data, given_action)

        df = pd.DataFrame(final_result, columns=['date', 'stock1', 'stock2', 'action', 'record', 'half_tax_profit',
                                                 'capital', 'predict_action_idx', 'correct_action_idx', 'same_action',
                                                 'two_action_same_profit'])
        df.to_csv(f'{MODEL_FOLDER_ROOT}/{folder_name}/second_stage/training_set_2_prediction.csv')
        second_training_label = [int(x) for x in list(df['half_tax_profit'] > 0)]

        second_loader_train, _ = DataTransfer(second_training_data, second_training_label)

        model_train(folder_name, second_loader_train, _, _, given_action, second_stage=True)

    return folder_name


if __name__ == '__main__':

    '''******************************* Do Your control below *******************************'''

    choose_mode = 'train and test' # or 'train' or 'test'

    two_stage = True

    test_start_year = 2018
    test_start_month = 1
    n_test_month = 19

    if two_stage:
        validate_start_year = test_start_year - 1
        second_stage_start_month = 3
        second_stage_months = 10
    else:
        validate_start_year = test_start_year - 1
        second_stage_months = 0

    validate_start_month = 1
    n_validate_months = 2
    n_training_months = 12

    '''******************************* Do Your control above *******************************'''




    if choose_mode == 'train':  # Training
        main(test_year=test_start_year, test_start_month=test_start_month, n_test_month=n_test_month,
             validate_start_year=validate_start_year, second_stage_months=second_stage_months, second_stage_start_month=second_stage_start_month,
             validate_start_month=validate_start_month, validate_months=n_validate_months,
             training_months=n_training_months,
             top_n=None, two_stage=True, load_pre_trained_model=False)

    elif choose_mode == 'test':  # Testing
        # must start with tmp7... and the final folder should have "first_stage" folder
        #model_folder = r'tmp7\2024-04-25-20-52-28-2018_350_None_None\\'  # 2018-201907
        #model_folder = r'tmp7\2024-08-06-21-57-43-2017_350_None_None\\' # 2017
        #model_folder = r"tmp7\2024-08-07-01-24-09-2018_350_None_None\\" # 2018
        #model_folder = r"tmp7\2024-08-07-18-49-25-2015_350_None_None\\" # 2015
        model_folder = r"tmp7\2024-08-07-22-21-13-2016_350_None_None\\" # 2016

        dump_testing_result(model_folder, only_first_stage=False)

    elif choose_mode == 'train and test':
        folder_name = main(test_year=test_start_year, test_start_month=test_start_month, n_test_month=n_test_month,
             validate_start_year=validate_start_year, second_stage_months=second_stage_months,
             validate_start_month=validate_start_month, validate_months=n_validate_months,
             training_months=n_training_months,
             top_n=None, two_stage=True, load_pre_trained_model=False)
        dump_testing_result(folder_name, only_first_stage=False)
