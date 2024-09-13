import pandas as pd
import os
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')
TRAIN_MODEL_ROOT_FOLDER = config['folder']['train_model_root_folder']
TOP_N_PAIR_IN_2015_2016 = list()
with open("../stock_info/HsinHuaChang_with_trend_TOP_N_STOCK1_IN_2013_2014.txt") as file:
    TOP_STOCK1 = [line.strip() for line in file]
with open("../stock_info/HsinHuaChang_with_trend_TOP_N_STOCK2_IN_2013_2014.txt") as file:
    TOP_STOCK2 = [line.strip() for line in file]
PAIR_NAME = list()
for a, b in zip(TOP_STOCK1, TOP_STOCK2):
    TOP_N_PAIR_IN_2015_2016.append(f'{a}_{b}')


def plot_performance_with_dd(daily_profit, daily_capital, metrics, dates, pdf_name='equity_curve', output_pdf=False):
    daily_open_number = metrics[0]
    daily_win_number = metrics[1]
    daily_normal_close_number = metrics[2]
    daily_stop_loss_number = metrics[3]
    total = np.cumsum(daily_profit)
    max_daily_capital = max(daily_capital)
    r = pd.DataFrame([x/max_daily_capital if max_daily_capital != 0 else 0 for x in daily_profit])
    sharpe_ratio = r.mean() / r.std() * np.sqrt(252)
    dd = list()
    tt = 0
    for i in range(len(daily_profit)):
        if i > 0 and total[i] > total[i-1]:
            tt = total[i]
        dd.append(total[i]-tt)
    try:
        xs = [datetime.datetime.strptime(str(d), '%Y%m%d').date() for d in dates]
        highest_x = []
        highest_dt = []
        for i in range(len(total)):
            if total[i] == max(total[:i+1]) and total[i] > 0:
                highest_x.append(total[i])
                highest_dt.append(i)
        mpl.style.use('seaborn')
        f, ax = plt.subplots(2, sharex='all', figsize=(20, 12), gridspec_kw={'height_ratios': [3, 1]})
        ax[0].plot(np.arange(len(xs)), total, color='b', zorder=1)
        ax[0].scatter(highest_dt, highest_x, color='lime', marker='o', s=40, zorder=2)
        ax[0].set_title('Pairs trading DQN Equity Curve, Tax cost={}, Cost gate={}'.format(0.0015, 0.0015),
                        fontsize=20)
        ax[1].bar(np.arange(len(xs)), dd, color='red')
        date_tickers = dates

        def format_date(x, pos=None):
            if x < 0 or x > len(date_tickers)-1:
                return ''
            return date_tickers[int(x)]
        ax[0].xaxis.set_major_locator(MultipleLocator(80))
        ax[0].xaxis.set_major_formatter(FuncFormatter(format_date))
        ax[0].grid(True)
        shift = (max(total)-min(total))/20
        text_loc = max(total)-shift*8
        ax[0].text(np.arange(len(xs))[5], text_loc, 'Total open number: %d' % daily_open_number, fontsize=15)
        ax[0].text(np.arange(len(xs))[5], text_loc-shift, 'Total profit: %0.2f' % total[-1], fontsize=15)
        ax[0].text(np.arange(len(xs))[5], text_loc - shift * 2, 'Win rate: %.4f' % (daily_win_number / daily_open_number),
                   fontsize=15)
        ax[0].text(np.arange(len(xs))[5], text_loc - shift * 3, 'Normal close rate: %.4f' %
                   (daily_normal_close_number / daily_open_number), fontsize=15)
        ax[0].text(np.arange(len(xs))[5], text_loc - shift * 4, 'Stop loss rate: %.4f' %
                   (daily_stop_loss_number / daily_open_number), fontsize=15)
        ax[0].text(np.arange(len(xs))[5], text_loc - shift * 5, 'Max drawdown: %d' % min(dd), fontsize=15)
        ax[0].text(np.arange(len(xs))[5], text_loc - shift * 6, 'sharpe ratio: %.2f' % sharpe_ratio, fontsize=15)
        plt.tight_layout()
        if output_pdf:
            pp = PdfPages(f'{TRAIN_MODEL_ROOT_FOLDER}/{sys.argv[1]}/{pdf_name}.pdf')
            pp.savefig(f)
            pp.close()
        # plt.show()
        plt.close('all')
    except Exception as e:
        print(f'Generate performance error! error: {e}')
    return sharpe_ratio, max_daily_capital, min(dd)


def dump_each_cluster_result(top_n=None):
    folder = sys.argv[1]  # cluster main folder
    df = pd.DataFrame()
    total_folder = next(os.walk(f'{TRAIN_MODEL_ROOT_FOLDER}/{folder}'))[1]
    if top_n is None:
        target_folder_idx = range(len(total_folder))
        # target_folder_idx = range(int(sys.argv[2]), int(sys.argv[3]))
    else:
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
        if not os.path.exists(f'{TRAIN_MODEL_ROOT_FOLDER}/{folder}/{cluster_folder}/Test/testing_result.csv'):
            print(f'execute python trained_dqn_agent.py {folder}/{cluster_folder} idx: {idx}')
            subprocess.run(['python', 'trained_dqn_agent.py', f'{folder}/{cluster_folder}'])
        else:
            print(f'{cluster_folder} already has testing result')
        # subprocess.run(['python', 'trained_dqn_agent.py', f'{folder}/{cluster_folder}'])
        try:
            tmp = pd.read_csv(f'{TRAIN_MODEL_ROOT_FOLDER}/{folder}/{cluster_folder}/Test/testing_result.csv')
        except FileNotFoundError:
            print(f'no result in {cluster_folder}!')
            continue
        if tmp.empty:
            print(f'{folder}/{cluster_folder} result is empty!')
            continue
        df = df.append(tmp)
    # print(f'filter only 2017 result!')
    # df = df[[str(dt).startswith('2017') for dt in df['date']]]
    df.sort_values(by=['date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def dump_final_result(df, top_n=None):
    total_profit = sum(df['half_tax_profit'])
    open_times = sum(df['record'] != 0)
    if open_times != 0:
        stop_loss_times = sum(df['record'] == -2)
        normal_close_times = sum(df['record'] == 666)
        force_exit_times = sum(df['record'] == -4)
        win_times = sum(df['half_tax_profit'] > 0)
        daily_profit = list(df.groupby(['date']).sum()['half_tax_profit'])
        non_zero_daily_profit_idx = np.array(daily_profit) != 0
        daily_profit = list(np.array(daily_profit)[non_zero_daily_profit_idx])
        daily_capital = list(df.groupby(['date']).sum()['capital'])
        metrics = [open_times, win_times, normal_close_times, stop_loss_times]
        dates = list(df.groupby(['date']).sum().index)
        dates = list(np.array(dates)[non_zero_daily_profit_idx])
        sharpe_ratio, max_daily_capital, mdd = plot_performance_with_dd(daily_profit, daily_capital, metrics, dates)
        trade_df = df[df['capital'] != 0]
        pair_return_list = pd.DataFrame(trade_df['half_tax_profit'] / trade_df['capital'])
        pair_based_sharpe_ratio = pair_return_list.mean() / pair_return_list.std()
        '''
        print format:
        Profit
        Win Rate/Normal Close Rate
        (Normal close number, Stop loss close number, Force close number, Total number)
        Sharpe ratio (daily-based)
        Sharpe ratio (pair-based)
        Profit of per open (thousand)/Trade capital (thousand)/MDD
        '''
        if top_n is not None:
            print(f'top {top_n} performance:')
        print(f'{total_profit:.2f}')
        print(f'{(win_times / open_times * 100):.2f}%/{(normal_close_times / open_times * 100):.2f}%')
        print(f'({normal_close_times}/{stop_loss_times}/{force_exit_times}/{open_times})')
        print(f'{sharpe_ratio[0]:.4f}')
        print(f'{pair_based_sharpe_ratio[0]:.4f}')
        print(f'{total_profit / open_times:.4f}/{max_daily_capital:.2f}/{mdd:.2f}')
    else:
        print(f'{0:.2f}')
        print(f'{0:.2f}%/{0:.2f}%')
        print(f'({0}/{0}/{0}/{0})')
        print(f'{0:.4f}')
        print(f'{0:.4f}')
        print(f'{0:.4f}/{0:.2f}/{0:.2f}')
    results = dict()
    results["Profit"] = total_profit
    results["Win Rate"] = win_times / open_times * 100 if open_times !=0 else 0
    results["Normal close rate"] = normal_close_times / open_times * 100 if open_times !=0 else 0
    results["Normal close number"] = normal_close_times if open_times !=0 else 0
    results["Stop loss close number"] = stop_loss_times if open_times !=0 else 0
    results["Force close number"] = force_exit_times if open_times !=0 else 0
    results["Total number"] = open_times
    results["Sharpe ratio (daily-based)"] = sharpe_ratio[0] if open_times !=0 else 0
    results["Sharpe ratio (pair-based)"] = pair_based_sharpe_ratio[0] if open_times !=0 else 0
    results["Profit of per open (thousand)"] = total_profit / open_times if open_times !=0 else 0
    results["Trade capital (thousand)"] = max_daily_capital if open_times !=0 else 0
    results["MDD"] = mdd if open_times !=0 else 0
    return results


def main():
    if len(sys.argv) == 3:
        top_n = int(sys.argv[2])
    else:
        top_n = None
    df = dump_each_cluster_result(top_n)
    dump_final_result(df, top_n)


if __name__ == '__main__':
    #main()
    root = r'C:\Users\eason\Desktop\PTDQN-main\ptdqn_model\tmp28\trend\\'
    df = pd.DataFrame()
    for f in os.listdir(root):
        print(root+f+"\\Test\\testing_result.csv")
        df = df.append(pd.read_csv(root+"\\"+f+"\\Test\\testing_result.csv"))
        print(df)
    results_summary = dump_final_result(df)
    pd.DataFrame(results_summary, index=[0]).to_csv(root+'testing_result_summary.csv')
