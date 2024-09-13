import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
import datetime
import os
import sys
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')
if config['folder']['root_folder'] not in sys.path:
    sys.path.insert(0, config['folder']['root_folder'])
from agent.dump_all_cluster_result import dump_final_result


MINPRICE_FOLDER = 'D:/HsinHuaChangThesis/thesis/data/min_data'
MODEL_FOLDER = 'D:/OPTICS_model'


def get_5min_price_return(dt, mode=0):
    df = pd.read_csv(f'{MINPRICE_FOLDER}/{dt[:4]}/minprice/{dt}_min_stock.csv')
    # Use formation period data to get 5min price return
    df = df.iloc[16:166, :]
    df.reset_index(drop=True, inplace=True)
    if mode == 0:
        # Get the 5-min close price
        df = df.iloc[range(4, 150, 5), :]
    elif mode == 1:
        # Get the 5-min average price
        df = df.rolling(window=5).mean().iloc[range(4, 150, 5), :]
    elif mode == 2:
        # Get the 5-min moving average price
        df = df.rolling(window=5).mean().iloc[4:, :]
    else:
        raise NotImplementedError('Not implement other mode!')
    return_df = df.pct_change().iloc[1:, :]
    return_df.reset_index(drop=True, inplace=True)
    return return_df


def dimension_reduction(stock_return, dim=5):
    pca = PCA(n_components=dim)
    new_stock_return = pca.fit_transform(stock_return.T)
    return new_stock_return


def unsupervised_learning(data):
    clustering = OPTICS().fit(data)
    return clustering.labels_


def dump_original_performance():
    # Dump the original performance as described in P.25 and P.27 of Hsin-Hua Chang thesis
    for t in ['', 'trend_stationary_']:
        for year in range(2015, 2019):
            file_name = f'{year}_{t}trading_result'
            df = pd.read_csv(f'{MODEL_FOLDER}/original_performance/{file_name}.csv')
            df.rename(columns={'Reward_015': 'half_tax_profit', 'Trade_capital': 'capital'}, inplace=True)
            print(file_name)
            dump_final_result(df)


def main(year, trend_stationary=False, mode=0):
    if not trend_stationary:
        file_name = f'{year}_trading_result'
    else:
        file_name = f'{year}_trend_stationary_trading_result'
    origin_performance = pd.read_csv(f'{MODEL_FOLDER}/original_performance/{file_name}.csv')
    origin_performance.rename(columns={'Reward_015': 'half_tax_profit', 'Trade_capital': 'capital'}, inplace=True)
    origin_performance['date'] = [dt.replace('-', '') for dt in origin_performance['date']]
    origin_performance['S1'] = origin_performance['S1'].astype('int').astype('str')
    origin_performance['S2'] = origin_performance['S2'].astype('int').astype('str')
    unique_dt = np.unique(origin_performance['date'])
    total_target_pair = pd.DataFrame()
    for dt in unique_dt:
        print(dt)
        all_stock_5min_return = get_5min_price_return(dt, mode=mode)
        stock_name = all_stock_5min_return.columns
        pca_stock_return = dimension_reduction(all_stock_5min_return)
        label = unsupervised_learning(pca_stock_return)
        if len(stock_name) != len(label):
            raise ValueError(f'Length doe not match {len(stock_name)} vs {len(label)}!')
        print(np.unique(label, return_counts=True))
        daily_pair_performance = origin_performance[origin_performance['date'] == dt]
        for cluster_idx in range(0, max(label)+1):
            print(f'process cluster idx: {cluster_idx}')
            same_cluster_stocks = stock_name[label == cluster_idx]
            daily_target_pair_idx = daily_pair_performance.apply(
                lambda r: r['S1'] in same_cluster_stocks and r['S2'] in same_cluster_stocks, axis=1)
            if sum(daily_target_pair_idx) > 0:
                total_target_pair = total_target_pair.append(daily_pair_performance[daily_target_pair_idx])
    now = datetime.datetime.now()
    date = now.strftime('%Y-%m-%d')
    if not os.path.exists(f'{MODEL_FOLDER}/{date}'):
        os.makedirs(f'{MODEL_FOLDER}/{date}')
    total_target_pair.to_csv(f'{MODEL_FOLDER}/{date}/{year}_optics_stationary_trading_result_{mode}.csv', index=False)
    print(f'{year} OPTICS with mode {mode} performance:')
    dump_final_result(total_target_pair)


if __name__ == '__main__':
    for y in range(2016, 2019):
        for m in range(3):
            main(y, mode=m)
