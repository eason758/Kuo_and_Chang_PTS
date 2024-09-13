import numpy as np
import json
import logging
import sys
from shutil import copyfile
import os
import pickle
import configparser

config = configparser.ConfigParser()
if os.path.isfile('config.ini'):
    config.read('config.ini')
else:
    # This module is called from agent
    config.read('../config.ini')
PTDQN_DATA_FOLDER = config['folder']['data_root_folder']
if config['folder']['root_folder'] + '/dump_json_data' not in sys.path:
    sys.path.insert(0, config['folder']['root_folder'] + '/dump_json_data')
from dump_json_data import dump_pairs_trading_reward_json  # noqa: E402


def choose_specific_stock_pair(data, stock1, stock2):
    data = np.array(data)
    idx = list()

    if type(stock1) == list:
        stock1 = [str(s) for s in stock1]
    else:
        stock1 = [str(stock1)]
    if type(stock2) == list:
        stock2 = [str(s) for s in stock2]
    else:
        stock2 = [str(stock2)]
    for i in range(len(data)):
        for s1, s2 in zip(stock1, stock2):
            if str(data[i]['stock1']) < str(data[i]['stock2']):
                if s1 == str(data[i]['stock1']) and s2 == str(data[i]['stock2']):
                    idx.append(i)
            else:
                if s2 == str(data[i]['stock1']) and s1 == str(data[i]['stock2']):
                    idx.append(i)
    return list(data[idx])



def choose_specific_model_type_pair(data, specific_model_type, model_type_json_file=None):
    data = np.array(data)
    if model_type_json_file is not None:
        print(f'loading {model_type_json_file}')
        logging.info(f'loading {model_type_json_file}')
        with open(f'{PTDQN_DATA_FOLDER}/{model_type_json_file}', 'r') as f:
            model_type_json = json.load(f)
    else:
        try:
            with open(f'{PTDQN_DATA_FOLDER}/model_type.json', 'r') as f:
                model_type_json = json.load(f)
        except FileNotFoundError:
            model_type_json = None
    idx = list()
    for i, d in enumerate(data):
        pair_key = d['date'] + '_' + d['stock1'] + '_' + d['stock2']
        if 'model_type' in d and model_type_json_file is None:
            if d['model_type'] == specific_model_type:
                idx.append(i)
        else:
            if model_type_json is None:
                raise ValueError('Given json file does not contain model type key!')
            if model_type_json and model_type_json[pair_key] == specific_model_type:
                idx.append(i)
    return list(data[idx])


def delete_irrelevant_action(data, needed_action):
    col = ['stock1', 'stock2', 'model_type', 'mu', 'stdev', 'spread', 'stock1_price', 'stock2_price', 'w1', 'w2', 'w0',
           'date']
    needed_action_key = [str(x[0]) + ',' + str(x[1]) for x in needed_action]
    needed_key = needed_action_key + col
    for i, d in enumerate(data):
        for k in list(d):
            if k not in needed_key:
                del d[k]
    return data


def filter_specific_data(data, stock1=None, stock2=None, specific_model_type=0, given_action=None,
                         model_type_json_file=None):
    if stock1 is not None and stock2 is not None:
        data = choose_specific_stock_pair(data, stock1, stock2)
    elif stock1 is None and stock2 is None:
        pass
    elif stock1 is None or stock2 is None:
        raise ValueError('stock1 and stock2 must given')
    if specific_model_type != 0:
        # if specific_model_type not in [1, 2, 3]:
        #     raise ValueError('Unknown model type!')
        data = choose_specific_model_type_pair(data, specific_model_type, model_type_json_file)
    if given_action is not None:
        data = delete_irrelevant_action(data, given_action)
    return data


def load_single_file_data(file_name, pickle_format, stock1, stock2, specific_model_type, given_action,
                          model_type_json_file):
    if pickle_format:
        with open(f'{file_name}.pickle', 'rb') as f:
            pairs_dict_temp = pickle.load(f)
    else:
        # load json file
        with open(f'{file_name}.json', 'r') as f:
            pairs_dict_temp = json.load(f)
    pairs_dict_temp = filter_specific_data(pairs_dict_temp, stock1=stock1, stock2=stock2,
                                           specific_model_type=specific_model_type, given_action=given_action,
                                           model_type_json_file=model_type_json_file)

    return pairs_dict_temp


def load_single_folder_data(data_folder, year, month, test_months, train_months, stock1=None, stock2=None,
                            testing=False, specific_model_type=0, given_action=None, model_type_json_file=None):
    if testing:
        logging_method = print
    else:
        logging_method = logging.info
    if data_folder.endswith('pickle'):
        pickle_format = True
    else:
        pickle_format = False
    train_pairs_data_dir = f'{PTDQN_DATA_FOLDER}/{data_folder}/Pairs'
    test_pairs_data_dir = f'{PTDQN_DATA_FOLDER}/{data_folder}/Pairs'
    logging_method('Loading test data from: {}'.format(test_pairs_data_dir))
    validate_data = []
    validate_year = year
    validate_month = month - 1
    for test_months in range(test_months):
        validate_month += 1
        if validate_month > 12:
            validate_month = 1
            validate_year += 1
            if validate_year > 2020:
                logging_method('Year load error')
                break
        logging_method("Add validate set year/month {} {}".format(validate_year, validate_month))
        file_name = test_pairs_data_dir + str(validate_year) + str(validate_month).zfill(2)
        pairs_dict_temp = load_single_file_data(file_name, pickle_format, stock1, stock2, specific_model_type,
                                                given_action, model_type_json_file)
        validate_data = validate_data + pairs_dict_temp
    train_data = []
    logging_method('Loading train data from: {}'.format(test_pairs_data_dir))
    for train_months in range(train_months):
        month -= 1
        if month < 1:
            month = 12
            year -= 1
        logging_method("Add train set year/month {} {}".format(year, month))
        file_name = train_pairs_data_dir + str(year) + str(month).zfill(2)
        pairs_dict_temp = load_single_file_data(file_name, pickle_format, stock1, stock2, specific_model_type,
                                                given_action, model_type_json_file)
        train_data = train_data + pairs_dict_temp
    logging_method('Total Pairs: {}'.format(len(train_data) + len(validate_data)))
    logging_method('Train Pairs: {}'.format(len(train_data)))
    logging_method('Test Pairs: {}'.format(len(validate_data)))
    return train_data, validate_data


def load_data(given_data_folder, year, month, test_months, train_months, stock1=None, stock2=None, testing=False,
              specific_model_type=0, given_action=None, model_type_json_file=None):
    """ Load training and testing data of pairs trading performance.

    Args:
        given_data_folder (str or list): Load json data from `given_data_folder` located in the `data_root_folder` of
                                         config.
        year (int): The start of the testing year.
        month (int): The start of the testing month.
        test_months (int): Number of testing months.
        train_months (int): Number of training months.
        stock1 (None or str or list): If not `None`, only load the pair for specific stock1.
        stock2 (None or str or list): If not `None`, only load the pair for specific stock2.
        testing (bool): If `False`, will use `logging.info` to print. Otherwise, will use original `print` method.
        specific_model_type (int): If not `None`, only load the pair for the given model type.
        given_action (None or list): If not `None`, only load specific given action and drop other irrelevant action.
        model_type_json_file (None or str): If not `None`, will load this json file located in the `data_root_folder`
                                            of config and use this json data to determine the type of each pair.
                                            Otherwise, will use original model type given in formation table.
    Examples:
        # Load testing data 201701~201702 and training data 201603~201612 from
        # f'{data_root_folder}/HsinHuaChang_reverse_1_new_std' \n
        >>> load_data('HsinHuaChang_reverse_1_new_std', 2017, 1, 2, 10, testing=True)
        Loading test data from: D:/ptdqn_data/Data/HsinHuaChang_reverse_1_new_std/Pairs
        Add validate set year/month 2017 1
        Add validate set year/month 2017 2
        Loading train data from: D:/ptdqn_data/Data/HsinHuaChang_reverse_1_new_std/Pairs
        Add train set year/month 2016 12
        Add train set year/month 2016 11
        Add train set year/month 2016 10
        Add train set year/month 2016 9
        Add train set year/month 2016 8
        Add train set year/month 2016 7
        Add train set year/month 2016 6
        Add train set year/month 2016 5
        Add train set year/month 2016 4
        Add train set year/month 2016 3
        Total Pairs: 116323
        Train Pairs: 97277
        Test Pairs: 19046
    Returns:
        tuple: (Number of testing training data, Number of testing data, training data, testing data)
    """
    if type(given_data_folder) == list:
        train_data = list()
        validate_data = list()
        for data_folder in given_data_folder:
            tmp_train, tmp_validate = load_single_folder_data(data_folder, year, month, test_months, train_months,
                                                              stock1=stock1, stock2=stock2, testing=testing,
                                                              specific_model_type=specific_model_type,
                                                              given_action=given_action,
                                                              model_type_json_file=model_type_json_file)
            train_data = train_data + tmp_train
            validate_data = validate_data + tmp_validate
    elif type(given_data_folder) == str:
        train_data, validate_data = load_single_folder_data(given_data_folder, year, month, test_months, train_months,
                                                            stock1=stock1, stock2=stock2, testing=testing,
                                                            specific_model_type=specific_model_type,
                                                            given_action=given_action,
                                                            model_type_json_file=model_type_json_file)
    else:
        raise ValueError('Given data folder type must be list or str!')
    return len(train_data), len(validate_data), train_data, validate_data


def get_optimal_actions_in_specific_pair(year, month, train_months, source_data_folder, stock1=None,
                                         stock2=None, testing=False, specific_model_type=0, folder_description=None,
                                         action_count=25, output_start_year=None, output_end_year=None,
                                         output_data=False):
    """ Using training data to get the optimal actions with `HighFreq` approach (Kuo, 2020)

    Args:
        year (int): The start of the training year.
        month (int): The start of the training month.
        train_months (int): Number of training months.
        source_data_folder (str or list): Load json data from `source_data_folder` located in the `data_root_folder` of
                                          config.
        stock1 (None or str or list): If not `None`, only load the pair for specific stock1.
        stock2 (None or str or list): If not `None`, only load the pair for specific stock2.
        testing (bool): If `False`, will use `logging.info` to print. Otherwise, will use original `print` method.
        specific_model_type (int): If not `None`, only load the pair for the given model type.
        folder_description (None or str): If not `None`, add this string to destination directory name.
        action_count (int): The number of target actions. Default: 25
        output_start_year (None or int): If not `None`, use this year to be the start year of dumping data.
                                         Otherwise the start year is 2013.
        output_end_year (None or int): If not `None`, use this year to be the end year of dumping data.
                                       Otherwise the end year is 2020.
        output_data (bool): If not `False`, will use `dump_pairs_trading_reward_json` to dump new performance file with
                     target actions and pairs.
    Returns:
        tuple: (data_folder, target_optimal_actions)
    """
    if type(source_data_folder) == str:
        with open(f'{PTDQN_DATA_FOLDER}/{source_data_folder}/agent_action_info.json', 'r') as f:
            given_action = json.load(f)
    elif type(source_data_folder) == list:
        given_action = None
        for data_folder in source_data_folder:
            with open(f'{PTDQN_DATA_FOLDER}/{data_folder}/agent_action_info.json', 'r') as f:
                if given_action is None:
                    given_action = json.load(f)
                else:
                    if given_action != json.load(f):
                        raise ValueError('Action does not match!')
    else:
        raise ValueError('Given data folder type must be list or str!')
    logging.info(f'Use {source_data_folder} to get {stock1} and {stock2} optimal action')
    total_max_profit_idx = list()
    month_count = 0
    total_pair_count = 0
    while month_count < train_months:
        start_year = year
        start_month = month - month_count
        while start_month < 1:
            start_year -= 1
            start_month += 12
        train_pairs_count, _, train_pairs_data, _ = load_data(source_data_folder, start_year, start_month, 0, 1,
                                                              stock1, stock2, testing, specific_model_type)
        total_pair_count += len(train_pairs_data)
        print(len(train_pairs_data))
        for i in range(len(train_pairs_data)):
            total_profit_list = list()
            for action in given_action:
                total_profit_list.append(train_pairs_data[i][f'{action[0]},{action[1]}']['half_tax_profit'])
            max_profit_idx = np.array(total_profit_list).argmax()
            total_max_profit_idx.append(max_profit_idx)
        month_count += 1
        del train_pairs_data
    logging.info(f'Use {len(total_max_profit_idx)} pairs to get optimal action')
    total_max_profit_idx = np.array(total_max_profit_idx)
    u, count = np.unique(total_max_profit_idx, return_counts=True)

    logging.info(f'unique actions: {u}')
    logging.info(f'unique actions count: {count}')
    count_sort_ind = np.argsort(-count)
    target_optimal_actions = np.array(given_action)[u[count_sort_ind[:action_count]]]
    target_optimal_actions = [list(x) for x in target_optimal_actions]
    print(target_optimal_actions)
    logging.info(f'target_optimal_actions: {target_optimal_actions}')
    if output_data:
        target_optimal_data_folder = \
            dump_pairs_trading_reward_json(target_optimal_actions, stock1, stock2, total_pair_count,
                                           folder_description=folder_description, start_year=output_start_year,
                                           end_year=output_end_year, overwrite=True, pickle_format=True)
        with open(f'{PTDQN_DATA_FOLDER}/{target_optimal_data_folder}/agent_action_info.json', 'r') as f:
            data_folder_actions = json.load(f)
        if data_folder_actions != target_optimal_actions:
            raise ValueError(f'Action does not match! Optimal actions are {target_optimal_actions}, '
                             f'data folder actions are {data_folder_actions}, data folder name is '
                             f'{target_optimal_data_folder}')
        return target_optimal_data_folder, target_optimal_actions
    else:
        return source_data_folder, target_optimal_actions


def dump_specific_pairs_to_pickle_with_source_json_data(stock1=None, stock2=None):
    source_data_folder = 'HsinHuaChang_reverse_random_300_new_std'
    with open(f'{PTDQN_DATA_FOLDER}/{source_data_folder}/agent_action_info.json', 'r') as f:
        given_action = json.load(f)
    if stock1 is not None and stock2 is not None:
        target_data_folder = \
            f'HsinHuaChang_reverse_random_{len(given_action)}_{len(stock1)}_{len(stock2)}_new_std_pickle'
    else:
        target_data_folder = \
            f'HsinHuaChang_reverse_random_{len(given_action)}_new_std_pickle'
    if not os.path.exists(f'{PTDQN_DATA_FOLDER}/{target_data_folder}'):
        os.makedirs(f'{PTDQN_DATA_FOLDER}/{target_data_folder}')
    else:
        pass
    copyfile(f'{PTDQN_DATA_FOLDER}/{source_data_folder}/agent_action_info.json',
             f'{PTDQN_DATA_FOLDER}/{target_data_folder}/agent_action_info.json')
    month_count = 0
    month = 1
    while month_count < 12 * 4:
        start_year = 2018
        start_month = month - month_count
        while start_month < 1:
            start_year -= 1
            start_month += 12
        train_pairs_count, _, train_pairs_data, _ = load_data(source_data_folder, start_year, start_month, 0, 1,
                                                              stock1, stock2, testing=True)
        dt = train_pairs_data[0]['date'][:6]
        with open(f'{PTDQN_DATA_FOLDER}/{target_data_folder}/Pairs{dt}.pickle', 'wb') as f:
            print(f'dump to {PTDQN_DATA_FOLDER}/{target_data_folder}/Pairs{dt}.pickle')
            pickle.dump(train_pairs_data, f)
        del train_pairs_data
        month_count += 1


if __name__ == '__main__':
    # DATA_FOLDER = 'HsinHuaChang_reverse_25_5_5_random300_new_std'
    # get_optimal_actions_in_specific_pair(2015, 11, 2, 22, DATA_FOLDER, testing=False)
    with open("stock_info/HsinHuaChang_with_trend_TOP_N_STOCK1_IN_2014_2015.txt") as F:
        TOP_STOCK1 = [line.strip() for line in F]
    with open("stock_info/HsinHuaChang_with_trend_TOP_N_STOCK2_IN_2014_2015.txt") as F:
        TOP_STOCK2 = [line.strip() for line in F]
    STOCK1 = TOP_STOCK1[:7342]
    STOCK2 = TOP_STOCK2[:7342]
    dump_specific_pairs_to_pickle_with_source_json_data()
