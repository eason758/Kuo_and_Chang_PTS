import sys
import torch
import torch.nn as nn
import torch.utils.data as Data
import json
import pandas as pd
import os
from dl_utlis import data_preprocess
from dump_all_cluster_result import dump_final_result
import numpy as np

MODEL_FOLDER_ROOT = 'D:/multi_scale_ResNet_model'


def dump_testing_result_csv(test_folder, df):
    redundant = ''
    if os.path.isfile(f'{test_folder}/testing_result.csv'):
        redundant = '-new'
    try:
        df.to_csv(f'{test_folder}/testing_result{redundant}.csv', index=False)
    except:
        raise ValueError(f'Can not create {test_folder}/testing_result{redundant}.csv')


def predict_label(trained_model, testing_data):
    trained_model.eval()
    action_list = list()
    whole_year = torch.FloatTensor(testing_data).cuda()
    torch_dataset_train = Data.TensorDataset(whole_year)
    whole_test = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=1024,
        shuffle=False,
    )
    for step, (batch_x,) in enumerate(whole_test):
        output = trained_model(batch_x)
        if output.shape == torch.Size([25]):
            print('Only one testing data! Expand the dimension!')
            output = torch.unsqueeze(output, 0)
        _, predicted = torch.max(output, 1)
        action_choose = predicted.cpu().numpy()
        action_choose = action_choose.tolist()
        action_list.append(action_choose)
    action_list = sum(action_list, [])
    return action_list


def get_performance_by_predicted_action(trained_model, testing_data, testing_label, testing_action_data, given_action):
    action_list = predict_label(trained_model, testing_data)
    print(f'Predict testing label len:{len(action_list)} testing data len:{len(testing_action_data)}')
    if len(action_list) != len(testing_action_data):
        raise ValueError(f'len(action_list): {len(action_list)} != len(testing_action_data): '
                         f'{len(testing_action_data)}')
    final_result = list()
    for predict_action_idx, data, correct_action_idx in zip(action_list, testing_action_data, testing_label):
        predict_action = given_action[predict_action_idx]
        predict_action_performance = data[f"{predict_action[0]},{predict_action[1]}"]
        correct_action = given_action[correct_action_idx]
        correct_action_performance = data[f"{correct_action[0]},{correct_action[1]}"]
        if predict_action_performance['half_tax_profit'] == correct_action_performance['half_tax_profit']:
            same_profit = 1
        else:
            same_profit = 0
        final_result.append([data['date'], data['stock1'], data['stock2'], predict_action,
                             predict_action_performance['record'], predict_action_performance['half_tax_profit'],
                             predict_action_performance['capital'], predict_action_idx, correct_action_idx,
                             int(predict_action_idx == correct_action_idx), same_profit])
    return final_result


def dump_acc_result(df):
    with open(f'D:/ptdqn_data/Data/Johansen_test_model_type_5.json', 'r') as f:
        model_type_json = json.load(f)
    df['model_type'] = df.apply(lambda row: model_type_json[f"{row['date']}_{row['stock1']}_{row['stock2']}"], axis=1)
    print('Testing Acc / Same profit Acc')
    print('Without non open action testing Acc / Without non open action same profit Acc')
    print('Only stationary pair testing Acc / Only stationary pair same profit Acc')
    print('Without non open action & Only stationary pair testing Acc / '
          'Without non open action & Only stationary pair same profit Acc')

    print(f'{(sum(df["same_action"]) / len(df)):.4f}', end=' / ')
    print(f'{(sum(df["two_action_same_profit"]) / len(df)):.4f}')

    tmp = df.loc[[x == '[1000000, 2500000]' for x in df['action']], :]
    non_open_action_idx_arr = np.unique(tmp['predict_action_idx'])
    if len(non_open_action_idx_arr) != 1:
        raise ValueError(f'The length of non_open_action_idx must be 1 but got {len(non_open_action_idx_arr)}')
    non_open_action_idx = non_open_action_idx_arr[0]
    without_non_open_action_df = df[df['correct_action_idx'] != non_open_action_idx]

    print(f'{(sum(without_non_open_action_df["same_action"]) / len(without_non_open_action_df)):.4f}', end=' / ')
    print(f'{(sum(without_non_open_action_df["two_action_same_profit"]) / len(without_non_open_action_df)):.4f}')

    only_stationary_pair_df = df[df['model_type'] <= 3]
    print(f'{(sum(only_stationary_pair_df["same_action"]) / len(only_stationary_pair_df)):.4f}', end=' / ')
    print(f'{(sum(only_stationary_pair_df["two_action_same_profit"]) / len(only_stationary_pair_df)):.4f}')

    non_open_action_only_stationary_pair_df = without_non_open_action_df[without_non_open_action_df['model_type'] <= 3]
    print(f'{(sum(non_open_action_only_stationary_pair_df["same_action"]) / len(non_open_action_only_stationary_pair_df)):.4f}', end=' / ')
    print(f'{(sum(non_open_action_only_stationary_pair_df["two_action_same_profit"]) / len(non_open_action_only_stationary_pair_df)):.4f}')

    non_open_action_df = df[df['correct_action_idx'] == non_open_action_idx]
    print(f"total pairs: {len(df)}, "
          f"the label is non-open pairs: {len(non_open_action_df)}, "
          f"the label is non-open pairs/total pairs: {(len(non_open_action_df) / len(df)) * 100:.2f}%, "
          f'final pairs: {len(without_non_open_action_df)}')


def dump_testing_result(folder_name, only_first_stage=False):
    # Use trained folder to dump the performance in testing period.
    test_folder = f'{MODEL_FOLDER_ROOT}/{folder_name}/Test'
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    if not os.path.isfile(f'{test_folder}/testing_result.csv'):
        with open(f'{MODEL_FOLDER_ROOT}/{folder_name}/test_info.json', 'r') as f:
            test_info = json.load(f)
        test_year = test_info['test_year']
        test_month = test_info['test_month']
        test_months = test_info['test_months']
        given_action = test_info['given_action']
        stock1 = test_info['stock1']
        stock2 = test_info['stock2']
        specific_model_type = test_info['specific_model_type']
        model_type_json_file = test_info['model_type_json_file']
        stock_price_data_folder = test_info['stock_price_data_folder']
        data_folder = test_info['data_folder']
        if 'two_stage' in test_info:
            two_stage = test_info['two_stage']
        else:
            two_stage = False
        _, _, testing_data, testing_label, testing_action_data = \
            data_preprocess(stock_price_data_folder, data_folder, test_year, test_month, test_months, 0,
                            given_action, stock1, stock2, specific_model_type, model_type_json_file, testing=True)
        if os.path.exists(f"{MODEL_FOLDER_ROOT}/{folder_name}/first_stage"):
            tmp = f'{folder_name}/first_stage'
        else:
            tmp = folder_name
        if os.path.isfile(f"{MODEL_FOLDER_ROOT}/{tmp}/best_win_rate.pkl"):
            trained_model = torch.load(f"{MODEL_FOLDER_ROOT}/{tmp}/best_win_rate.pkl")
        else:
            print(f'{MODEL_FOLDER_ROOT}/{tmp}/best_win_rate.pkl does not exist! '
                  f'Use final.pkl instead!')
            trained_model = torch.load(f"{MODEL_FOLDER_ROOT}/{tmp}/final.pkl")
        # Use first stage to get each pair action
        final_result = get_performance_by_predicted_action(trained_model, testing_data, testing_label,
                                                           testing_action_data, given_action)
        df = pd.DataFrame(final_result, columns=['date', 'stock1', 'stock2', 'action', 'record', 'half_tax_profit',
                                                 'capital', 'predict_action_idx', 'correct_action_idx', 'same_action',
                                                 'two_action_same_profit'])
        if two_stage:
            trained_model = torch.load(f"{MODEL_FOLDER_ROOT}/{folder_name}/second_stage/final.pkl")
            win_loss_label = predict_label(trained_model, testing_data)
            df['second_stage_label'] = win_loss_label
        dump_testing_result_csv(test_folder, df)
    print(folder_name)
    result_df = pd.read_csv(f'{test_folder}/testing_result.csv')
    if not only_first_stage and 'second_stage_label' in list(result_df):
        print('Adopt two stage! Drop the pairs with loss label!')
        print('The performance of two stage!')
        result_df = result_df[result_df['second_stage_label'] == 1]
    else:
        print('The performance of single stage!')
    dump_acc_result(result_df)
    dump_final_result(result_df)


def agg_all_cluster_result(folder, top_n=None):
    df = pd.DataFrame()
    total_folder = next(os.walk(f'{MODEL_FOLDER_ROOT}/{folder}'))[1]
    if top_n is None:
        target_folder_idx = range(len(total_folder))
        # target_folder_idx = range(int(sys.argv[2]), int(sys.argv[3]))
    else:
        TOP_N_PAIR_IN_2015_2016 = list()
        with open("../stock_info/HsinHuaChang_with_trend_TOP_N_STOCK1_IN_2015_2016.txt") as file:
            TOP_STOCK1 = [line.strip() for line in file]
        with open("../stock_info/HsinHuaChang_with_trend_TOP_N_STOCK2_IN_2015_2016.txt") as file:
            TOP_STOCK2 = [line.strip() for line in file]
        for a, b in zip(TOP_STOCK1, TOP_STOCK2):
            TOP_N_PAIR_IN_2015_2016.append(f'{a}_{b}')
        target_folder_idx = list()
        for pair in TOP_N_PAIR_IN_2015_2016[:top_n]:
            stock1 = pair.split('_')[0]
            stock2 = pair.split('_')[1]
            if sum([(f'{stock1}_{stock2}' in x) for x in total_folder]) == 0:
                raise ValueError(f'can not find {pair} in given folder')
            if sum([(f'{stock1}_{stock2}' in x) for x in total_folder]) > 1:
                raise ValueError(f'find {pair} more 1 folder in given folder')
            target_folder_idx.append([(f'{stock1}_{stock2}' in x) for x in total_folder].index(True))
    for idx in target_folder_idx:
        cluster_folder = total_folder[idx]
        print(f'Load {MODEL_FOLDER_ROOT}/{folder}/{cluster_folder}/Test/testing_result.csv')
        tmp = pd.read_csv(f'{MODEL_FOLDER_ROOT}/{folder}/{cluster_folder}/Test/testing_result.csv')
        if tmp.empty:
            print(f'{folder}/{cluster_folder} result is empty!')
            continue
        df = df.append(tmp)
    df.sort_values(by=['date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    dump_acc_result(df)
    dump_final_result(df, top_n=top_n)


def main(folder, top_n=None, only_first_stage=False):
    df = pd.DataFrame()
    total_folder = next(os.walk(f'{MODEL_FOLDER_ROOT}/{folder}'))[1]
    if top_n is None:
        target_folder_idx = range(len(total_folder))
    else:
        raise NotImplementedError('Not implement yet!')
    for idx in target_folder_idx:
        cluster_folder = total_folder[idx]
        print(f'Load {MODEL_FOLDER_ROOT}/{folder}/{cluster_folder}/Test/testing_result.csv')
        tmp = pd.read_csv(f'{MODEL_FOLDER_ROOT}/{folder}/{cluster_folder}/Test/testing_result.csv')
        if tmp.empty:
            print(f'{folder}/{cluster_folder} result is empty!')
            continue
        df = df.append(tmp)
    df.sort_values(by=['date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if not only_first_stage and 'second_stage_label' in list(df):
        print('Adopt two stage! Drop the pairs with loss label!')
        print('The performance of two stage!')
        df = df[df['second_stage_label'] == 1]
    else:
        print('The performance of single stage!')
    dump_acc_result(df)
    dump_final_result(df)


if __name__ == '__main__':
    main(sys.argv[1], only_first_stage=True)
    main(sys.argv[1], only_first_stage=False)
