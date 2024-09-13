import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sys
import json
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')
if config['folder']['root_folder'] not in sys.path:
    sys.path.insert(0, config['folder']['root_folder'])
from utils import load_data
DATA_ROOT_FOLDER = config['folder']['data_root_folder']
TRAIN_MODEL_ROOT_FOLDER = config['folder']['train_model_root_folder']


def get_metric(action, train_data, trading_threshold=0.0015, trading_cost=0.0015):
    if trading_cost == 0.0015:
        profit_name = 'half_tax_profit'
    elif trading_cost == 0:
        profit_name = 'zero_tax_profit'
    else:
        raise ValueError('Unknown trading cost error!')
    action_name = str(action[0]) + ',' + str(action[1])
    profit = 0
    total_open_num = 0
    total_record_open_num = [0, 0, 0, 0]
    daily_capital = 0
    total_daily_capital = []
    daily_profit = 0
    total_daily_profit = []
    dt = None
    pair_return_list = []
    dt_list = list()
    for i in range(len(train_data)):
        if dt is None:
            dt = train_data[i]['date']
            dt_list.append(dt)
        if dt != train_data[i]['date']:
            total_daily_capital.append(daily_capital)
            daily_capital = 0
            total_daily_profit.append(daily_profit)
            daily_profit = 0
            dt = train_data[i]['date']
            dt_list.append(dt)
        if train_data[i]['stdev'] * action[0] <= trading_threshold:
            continue
        profit += train_data[i][action_name][profit_name]
        daily_profit += train_data[i][action_name][profit_name]
        daily_capital += train_data[i][action_name]['capital']
        if train_data[i][action_name]['capital'] != 0:
            pair_return_list.append(train_data[i][action_name][profit_name]/train_data[i][action_name]['capital'])
        if train_data[i][action_name][profit_name] > 0:
            total_record_open_num[3] += 1
        if train_data[i][action_name]['record'] in [666, -2, -4]:
            total_open_num += 1
            if train_data[i][action_name]['record'] == 666:
                total_record_open_num[0] += 1
            if train_data[i][action_name]['record'] == -2:
                total_record_open_num[1] += 1
            if train_data[i][action_name]['record'] == -4:
                total_record_open_num[2] += 1
    total_daily_capital.append(daily_capital)
    total_daily_profit.append(daily_profit)
    # if 0 in pair_return_list:
    #    raise ValueError('pair return can not be zero!')
    return profit, total_open_num, total_record_open_num, total_daily_capital, total_daily_profit, pair_return_list, \
        dt_list


def get_optimal_action(year, data_folder, trading_threshold=0.0015, trading_cost=0.0015, stock1=None, stock2=None,
                       given_action=None, train_months=24):
    """Use training data to get the optimal action which profit or sharpe ratio is the highest among all actions

    Args:
        year: The start of the testing year
        data_folder (str or list): Load json data from `given_data_folder` located in the `data_root_folder` of
                                   config.
        trading_threshold (0.0015): Trading cost gate
        trading_cost (0.0015):  Trading cost
        stock1 (None or list): If not `None`, only load the pair for specific stock1.
        stock2 (None or list): If not `None`, only load the pair for specific stock2.
        given_action (None or list): If not `None`, only load specific given action and drop other irrelevant action.
        train_months: Number of training months.
    Returns:
        tuple: (optimal sharpe ratio action index, optimal profit action index)
    """
    _, _, train_data, _ = load_data(data_folder, year, 1, test_months=0, train_months=train_months, testing=True,
                                    stock1=stock1, stock2=stock2, given_action=given_action)
    profit_list = []
    sharpe_ratio_list = []
    open_number = []
    performance_list = []
    for a in given_action:
        profit, total_open_num, total_record_open_num, total_daily_capital, total_daily_profit, pair_return_list, \
            dt_list = get_metric(a, train_data, trading_threshold=trading_threshold, trading_cost=trading_cost)
        max_daily_capital = max(total_daily_capital)
        non_zero_daily_profit_idx = np.array(total_daily_profit) != 0
        total_daily_profit = list(np.array(total_daily_profit)[non_zero_daily_profit_idx])
        r = pd.DataFrame([x/max_daily_capital if max_daily_capital != 0 else 0 for x in total_daily_profit])
        sharpe_ratio = r.mean() / r.std() * np.sqrt(252)
        profit_list.append(profit)
        if len(sharpe_ratio) == 0:
            sharpe_ratio_list.append(-9999)
        else:
            sharpe_ratio_list.append(sharpe_ratio[0])
        open_number.append(total_open_num)
        performance_list.append([x/total_open_num if total_open_num != 0 else 0 for x in total_record_open_num])
        print(f'{a[0]} {a[1]} profit: {profit}')
        print(f'{total_record_open_num[0]}/{total_open_num}')
        if total_open_num != 0:
            print(f'{total_record_open_num[0]/total_open_num}')
        else:
            print(f'{total_record_open_num[0]}/{total_open_num}')
        print(f'{sharpe_ratio_list[-1]}')
    print(f'optimal sharpe ratio action in {len(sharpe_ratio_list)} action:')
    optimal_sharpe_ratio_action = np.argmax(np.array(sharpe_ratio_list).flatten())
    print(max(np.array(sharpe_ratio_list).flatten()), given_action[optimal_sharpe_ratio_action])
    print(f'optimal profit action in {len(profit_list)} action:')
    optimal_profit_action = np.argmax(np.array(profit_list).flatten())
    print(max(np.array(profit_list).flatten()), given_action[optimal_profit_action])
    return optimal_sharpe_ratio_action, optimal_profit_action


def print_specific_action_performance(year, specific_action, data_folder, trading_threshold=0.0015,
                                      trading_cost=0.0015, stock1=None, stock2=None, given_action=None, test_months=24):
    """ Dump the performance of the specific action

    Args:
        year: The start of the testing year
        specific_action: specific action index
        data_folder (str or list): Load json data from `given_data_folder` located in the `data_root_folder` of
                                   config.
        trading_threshold (0.0015): Trading cost gate
        trading_cost (0.0015):  Trading cost
        stock1 (None or list): If not `None`, only load the pair for specific stock1.
        stock2 (None or list): If not `None`, only load the pair for specific stock2.
        given_action (None or list): If not `None`, only load specific given action and drop other irrelevant action.
        test_months (int): Number of testing months.
    Returns:
        print the performance
        format:
            Profit
            Win Rate/Normal Close Rate
            (Normal close number, Stop loss close number, Force close number, Total number)
            Sharpe ratio (daily-based)
            Sharpe ratio (pair-based)
            Profit of per open (thousand)/Trade capital (thousand)/MDD
    """
    _, _, _, test_data = load_data(data_folder, year, 1, train_months=0, test_months=test_months, testing=True,
                                   stock1=stock1, stock2=stock2, given_action=given_action)
    print(len(test_data))
    profit, total_open_num, total_record_open_num, total_daily_capital, total_daily_profit, pair_return_list, \
        dt_list = get_metric(given_action[specific_action], test_data, trading_threshold=trading_threshold,
                             trading_cost=trading_cost)
    max_daily_capital = max(total_daily_capital)
    non_zero_daily_profit_idx = np.array(total_daily_profit) != 0
    total_daily_profit = list(np.array(total_daily_profit)[non_zero_daily_profit_idx])
    dt_list = list(np.array(dt_list)[non_zero_daily_profit_idx])
    # print(total_daily_profit)
    # print(dt_list)
    r = pd.DataFrame([x/max_daily_capital if max_daily_capital != 0 else 0 for x in total_daily_profit])
    sharpe_ratio = r.mean() / r.std() * np.sqrt(252)
    money_per_trade = profit/total_open_num
    total = np.cumsum(total_daily_profit)
    dd = list()
    tt = 0
    for i in range(len(total_daily_profit)):
        if i > 0 and total[i] > total[i-1]:
            tt = total[i]
        dd.append(total[i]-tt)
    mdd = min(dd)
    r = pd.DataFrame(pair_return_list)
    pair_based_sharpe_ratio = r.mean() / r.std()
    print(data_folder)
    print(f'{given_action[specific_action]}')
    print(f'{profit:.2f}')
    print(f'{total_record_open_num[3]/total_open_num*100:.2f}%/{total_record_open_num[0]/total_open_num*100:.2f}%')
    print(f'({total_record_open_num[0]},{total_record_open_num[1]},{total_record_open_num[2]},{total_open_num})')
    print(f'{sharpe_ratio[0]:.4f}')
    print(f'{pair_based_sharpe_ratio[0]:.4f}')
    print(f'{money_per_trade:.4f}/{max_daily_capital:.2f}/{mdd:.2f}')


def get_optimal_actions_in_pair():
    data_folder = \
        'trend_stationary_20210614_tax_cost=0.0015_tax_cost_gate=0.0015_new_data_new_std_get_optimal_25_actions'
    total_year = range(2010, 2013)
    total_max_profit_idx = list()
    actions = None
    for year in total_year:
        for month in range(1, 13):
            given_action, _, test_data = load_data(data_folder, year, month, test_months=1, train_months=0)
            actions = given_action
            print(len(test_data))
            for i in range(len(test_data)):
                total_profit_list = list()
                for action in given_action:
                    total_profit_list.append(test_data[i][f'{action[0]},{action[1]}']['half_tax_profit'])
                max_profit_idx = np.array(total_profit_list).argmax()
                total_max_profit_idx.append(max_profit_idx)
            print(f'max profit list len: {len(total_max_profit_idx)}')
    total_max_profit_idx = np.array(total_max_profit_idx)
    u, count = np.unique(total_max_profit_idx, return_counts=True)
    count_sort_ind = np.argsort(-count)
    np.save('trend_stationary_25_actions_method3', np.array(actions)[u[count_sort_ind[:25]]])
    np.save('total_max_profit_idx', total_max_profit_idx)
    tmp = np.array(actions)[total_max_profit_idx]
    k_means_center = KMeans(n_clusters=25, random_state=555).fit(tmp).cluster_centers_
    np.save('trend_stationary_25_actions_method0', k_means_center)


if __name__ == '__main__':
    DATA_FOLDER = 'HsinHuaChang_reverse_random_300_4211_4211_new_std_pickle'
    rank = int(sys.argv[1])
    if rank == 29:
        source_folder = '2022/20220314-P118C2/2022-03-10-01-14-07-2017_20029_29_150'
    elif rank == 63:
        source_folder = '2022/20220314-P118C2/2022-03-10-01-27-17-2017_20063_63_150'
    elif rank == 127:
        source_folder = '2022/20220314-P118C2/2022-03-10-01-22-29-2017_200127_127_150'
    elif rank == 237:
        source_folder = '2022/20220314-P118C2/2022-03-10-01-22-59-2017_200237_237_150'
    elif rank == 425:
        source_folder = '2022/20220314-P118C2/2022-03-10-01-23-26-2017_200425_425_150'
    elif rank == 732:
        source_folder = '2022/20220314-P118C2/2022-03-10-02-50-23-2017_200732_732_150'
    elif rank == 1227:
        source_folder = '2022/20220314-P118C2/2022-03-10-05-01-42-2017_2001227_1227_150'
    elif rank == 1909:
        source_folder = '2022/20220314-P118C2/2022-03-10-02-19-27-2017_2001909_1909_150'
    elif rank == 2869:
        source_folder = '2022/20220314-P118C2/2022-03-10-01-58-25-2017_2002869_2869_150'
    elif rank == 4211:
        source_folder = '2022/20220314-P118C2/2022-03-10-01-49-28-2017_2004211_4211_150'
    else:
        raise ValueError('Unknown rank!')
    with open(f'{TRAIN_MODEL_ROOT_FOLDER}/{source_folder}/agent_info.json', 'r') as f:
        INFO = json.load(f)
        GIVEN_ACTION = INFO['GIVEN_ACTION']
    with open("../stock_info/HsinHuaChang_TOP_N_STOCK1_IN_2015_2016.txt") as f:
        TOP_STOCK1 = [line.strip() for line in f]
    with open("../stock_info/HsinHuaChang_TOP_N_STOCK2_IN_2015_2016.txt") as f:
        TOP_STOCK2 = [line.strip() for line in f]
    STOCK1 = TOP_STOCK1[:rank]
    STOCK2 = TOP_STOCK2[:rank]
    specific_year = 2017
    print(f'{rank}: top 25 freq action is {GIVEN_ACTION}')
    sharpe_ratio_action, profit_action = get_optimal_action(specific_year, DATA_FOLDER, given_action=GIVEN_ACTION
                                                            , stock1=STOCK1, stock2=STOCK2)
    print_specific_action_performance(specific_year, profit_action, DATA_FOLDER, given_action=GIVEN_ACTION
                                      , stock1=STOCK1, stock2=STOCK2)
    print(f'{rank}: top 25 freq action is {GIVEN_ACTION}')
    # get_optimal_actions_in_pair()
