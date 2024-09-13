import os
import numpy as np
import json
import matplotlib.pyplot as plt
import gym
import plotter
import pandas as pd
import logging


def pnl_calculate(payoff, rate):
    tax_price = payoff * (1 - rate * (payoff > 0))
    return tax_price


class PTDQN(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, folder, train_data, test_data, **kwargs):
        """
        OpenAI override function
        Load data and set action and observation space
        """
        try:
            with open('D:/ptdqn_data/Data/industry_code.json', 'r') as file:
                self.industry_code = json.load(file)
        except FileNotFoundError:
            self.industry_code = None
        try:
            with open('D:/ptdqn_data/Data/stock_industry.json', 'r') as file:
                self.stock_industry = json.load(file)
        except FileNotFoundError:
            self.stock_industry = None
        self.folder = folder
        self.test_folder = folder
        self.data = train_data
        self.test_data = test_data
        self.formation_period = kwargs.get('formation_period', 150)
        self.trading_period = kwargs.get('trading_period', 100)
        self.validate = False
        self.rewards = []
        self.done = False
        self.action = None
        self.pair_idx = 0
        self.t_idx = self.formation_period
        self.reward = 0  # the reward in an episode
        self.epoch_reward = 0  # the reward in each episode
        self.observation = None
        self.total_open_times = 0
        self.stop_loss_times = 0
        self.normal_close_times = 0
        self.win_times = 0
        self.half_cost_win_times = 0
        self.zero_cost_win_times = 0
        self.force_exit_close_times = 0
        self.state_length = kwargs.get('state_length', 150)
        self.dump = kwargs.get('dump', False)
        self.standardization = kwargs.get('standardization', True)
        self.testing = kwargs.get('testing', False)
        self.cost_gate = kwargs.get('cost_gate', 0)
        self.index_data = kwargs.get('index_data', None)
        self.stock_daily_return_data = kwargs.get('stock_daily_return_data', None)
        assert self.index_data is None or self.stock_daily_return_data is None, \
            "index daily return and stock daily return can't occur in the same time."
        input_shape = self.state_length  # set the input shape depending on the window size
        self.given_action = kwargs.get('given_action', None)
        self.stock_price_return = kwargs.get('stock_price_return', False)
        if self.given_action is None:
            raise ValueError('action space not given')
        self.action_space = gym.spaces.Discrete(len(self.given_action))
        print('Total state shape: ', input_shape)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(input_shape,))
        self.daily_pnl = 0
        self.daily_pnl_list = list()
        self.daily_pnl_half_tax = 0
        self.daily_pnl_half_tax_list = list()
        self.daily_pnl_zero_tax = 0
        self.daily_pnl_zero_tax_list = list()
        self.date_list = list()
        self.date = None
        self.daily_capital = 0
        self.pair_return_list = list()
        self.pair_half_tax_return_list = list()
        self.pair_zero_tax_return_list = list()
        self.daily_capital_list = list()
        self.pnl = 0
        self.pnl_zero_tax = 0
        self.pnl_half_tax = 0
        self.trade_type_list = list()
        self.t_idx_list = list()
        self.action_list = list()
        self.pair_pnl = list()
        self.step_count = 0
        self.total_action = list()
        self.less_threshold_count = 0
        self.training_log = []
        self.testing_log = []
        self.specific_stock = kwargs.get('specific_stock', None)
        self.fixed_action = kwargs.get('fixed_action', None)
        self.only_twse = kwargs.get('only_twse', False)
        self.only_stock_price_ln_return = kwargs.get('only_stock_price_ln_return', False)
        self.only_stock_price = kwargs.get('only_stock_price', False)
        self.stock_price_ln_return = kwargs.get('stock_price_ln_return', False)
        self.model_name = kwargs.get('model_name', None)
        # self.ignore_pair = np.load('ignore_pair.npy')
        self.ignore_pair = None
        self.reset()

    def step(self, action):
        """
        OpenAI override function
        One step in the environment means:
        1) take action: move agent to next position and make a trade action
        2) store the action and the new value
        3) get reward
        return: the new state, reward and if the data are done
        """
        self.action = self.given_action[action]
        if self.fixed_action is not None:
            self.action = self.given_action[self.fixed_action]

        # print(action)
        self.step_count += 1
        if self.testing or self.validate:
            if self.pair_idx == len(self.data):
                # self.daily_pnl_reset()
                self.done = True
                self.reward = 0
                return self.observation, self.reward, self.done, {}
        # If the are still more pairs data to process
        if self.pair_idx < len(self.data):
            self.reward, close_timing = self.get_known_pt_reward()
            self.epoch_reward += self.reward
            if close_timing == -999:
                # No open portfolio
                self.t_idx = self.formation_period + self.trading_period
            else:
                self.t_idx = self.formation_period + close_timing

            self.observation = self.input_s()
            self.done = True
        else:
            raise ValueError('pair_idx is out of range')
        if self.done:
            if self.dump:
                self.plot_spread()
            self.trade_type_list = list()
            self.t_idx_list = list()
            self.action_list = list()
            self.pair_pnl = list()
            self.step_count = 0
            if self.pair_idx == len(self.data) - 1:
                self.epoch_reset()
            else:
                self.pair_idx += 1
                if not self.testing and self.ignore_pair:
                    while self.pair_idx <= len(self.data) - 1:
                        name = self.data[self.pair_idx]['date'] + '_' + self.data[self.pair_idx]['stock1'] + '_' + \
                               self.data[self.pair_idx]['stock2']
                        if name in self.ignore_pair:
                            logging.info(f'{self.pair_idx} {name} is ignored.')
                            self.pair_idx += 1
                        else:
                            break
                    if self.pair_idx > len(self.data) - 1:
                        self.epoch_reset()
        return self.observation, self.reward, self.done, {}

    def epoch_reset(self):
        # if not os.path.exists(self.folder + '/csv'):
        #    os.makedirs(self.folder + '/csv')
        if not self.testing:
            """
            pd.DataFrame(self.training_log, columns=['date', 'stock1', 'stock2', 'action', 'record', 'reward',
                                                     'half_tax_profit']).to_csv(f'{self.folder}/csv/'
                                                                                f'{len(self.rewards)}.csv')
            """
            self.training_log = []
        self.daily_pnl_reset()
        self.rewards.append(self.epoch_reward)
        print('EPOCH:{}'.format(len(self.rewards)))
        if len(self.rewards) % 50 == 0 and not self.testing:
            # self.plot_train_rewards(title='{}_train_rewards'.format(len(self.rewards)))
            print(self.rewards)
        self.epoch_reward = 0
        self.pair_idx = 0

    def reset(self):
        """
        After each epoch or at a start of a process (train, test, validation) reset the variables.
        """
        if self.testing:
            self.data = self.test_data
            self.test_folder = self.folder + '/Test'
            if not os.path.exists(self.test_folder):
                os.makedirs(self.test_folder)
        self.t_idx = self.formation_period
        self.action = None
        self.done = False
        self.observation = self.input_s()
        return self.observation

    def render(self, mode='human', close=False):
        """
        Gym function called at the end of a process.
        """
        super().reset()

    def metrics_reset(self):
        self.pnl = 0
        self.pnl_half_tax = 0
        self.pnl_zero_tax = 0
        self.pair_idx = 0
        self.total_open_times = 0
        self.stop_loss_times = 0
        self.normal_close_times = 0
        self.force_exit_close_times = 0
        self.t_idx = self.formation_period
        self.win_times = 0
        self.half_cost_win_times = 0
        self.zero_cost_win_times = 0
        self.daily_pnl = 0
        self.daily_pnl_list = list()
        self.daily_pnl_half_tax = 0
        self.daily_pnl_half_tax_list = list()
        self.daily_pnl_zero_tax = 0
        self.daily_pnl_zero_tax_list = list()
        self.date_list = list()
        self.daily_capital = 0
        self.daily_capital_list = list()
        self.date = None
        self.pair_return_list = list()
        self.pair_half_tax_return_list = list()
        self.pair_zero_tax_return_list = list()
        self.total_action = list()

    def daily_pnl_reset(self):
        if self.testing:
            self.daily_pnl_list.append(self.daily_pnl)
            self.daily_pnl = 0
            self.daily_pnl_half_tax_list.append(self.daily_pnl_half_tax)
            self.daily_pnl_half_tax = 0
            self.daily_pnl_zero_tax_list.append(self.daily_pnl_zero_tax)
            self.daily_pnl_zero_tax = 0
            self.date_list.append(self.date)
            self.date = self.data[self.pair_idx]['date']
            self.daily_capital_list.append(self.daily_capital)
            self.daily_capital = 0

    def input_s(self):
        """
        Prepare the input to the agent
        """
        pair_idx = self.pair_idx
        t_idx = self.t_idx
        if self.standardization:
            if self.index_data is not None:
                # state: spread + daily stock index return
                if self.testing:
                    print('input state: spread + daily stock index return')
                dt = self.data[pair_idx]['date']
                observation = \
                    (np.asarray(self.data[pair_idx]['spread'][t_idx - 150:t_idx]) -
                     self.data[pair_idx]['mu']) / self.data[pair_idx]['stdev']
                stock1_code = self.industry_code[self.stock_industry[self.data[pair_idx]['stock1']]]
                stock2_code = self.industry_code[self.stock_industry[self.data[pair_idx]['stock2']]]
                if self.only_twse:
                    stock1_code = 'T00'
                    stock2_code = 'T00'
                    # print('only twse')
                observation = np.append(observation, self.index_data[stock1_code][dt])
                observation = np.append(observation, self.index_data[stock2_code][dt])
            elif self.stock_daily_return_data is not None:
                # state: spread + daily stock return
                if self.testing:
                    print('input state: spread + daily stock return')
                dt = self.data[pair_idx]['date']
                observation = \
                    (np.asarray(self.data[pair_idx]['spread'][t_idx - 150:t_idx]) -
                     self.data[pair_idx]['mu']) / self.data[pair_idx]['stdev']
                observation = np.append(observation, self.stock_daily_return_data[self.data[pair_idx]['stock1']][dt])
                observation = np.append(observation, self.stock_daily_return_data[self.data[pair_idx]['stock2']][dt])
            else:
                if self.state_length == 450:
                    observation = \
                        (np.asarray(self.data[pair_idx]['spread'][t_idx - 150:t_idx]) -
                         self.data[pair_idx]['mu']) / self.data[pair_idx]['stdev']
                    if self.stock_price_ln_return:
                        if self.testing:
                            print('input state: spread + stock price ln return')
                        stock1_return = np.log(np.array(self.data[pair_idx]['stock1_price'])[1:] /
                                               np.array(self.data[pair_idx]['stock1_price'])[:-1])
                        stock1_return = np.insert(stock1_return, 0, 0)
                        stock2_return = np.log(np.array(self.data[pair_idx]['stock2_price'])[1:] /
                                               np.array(self.data[pair_idx]['stock2_price'])[:-1])
                        stock2_return = np.insert(stock2_return, 0, 0)
                        observation = np.append(observation, stock1_return[t_idx - 150:t_idx])
                        observation = np.append(observation, stock2_return[t_idx - 150:t_idx])
                    elif not self.stock_price_return:
                        # state: spread + intra day stock price
                        if self.testing:
                            print('input state: spread + intra day stock price')
                        stock1_price = np.asarray(self.data[pair_idx]['stock1_price'][t_idx - 150:t_idx])
                        if np.std(stock1_price) == 0:
                            observation = np.append(observation, (stock1_price - np.mean(stock1_price)) / 1)
                        else:
                            observation = np.append(
                                observation, (stock1_price - np.mean(stock1_price)) / np.std(stock1_price))
                        stock2_price = np.asarray(self.data[pair_idx]['stock2_price'][t_idx - 150:t_idx])
                        if np.std(stock2_price) == 0:
                            observation = np.append(observation, (stock2_price - np.mean(stock2_price)) / 1)
                        else:
                            observation = np.append(
                                observation, (stock2_price - np.mean(stock2_price)) / np.std(stock2_price))
                    else:
                        # state: spread + intra day stock price return
                        if self.testing:
                            print(f'{self.pair_idx} input state: spread + intra day stock price return')
                        stock1_return = \
                            (np.array(self.data[pair_idx]['stock1_price'])[1:] -
                             np.array(self.data[pair_idx]['stock1_price'])[:-1]) / \
                            np.array(self.data[pair_idx]['stock1_price'])[:-1]
                        stock1_return = np.insert(stock1_return, 0, 0)

                        stock2_return = \
                            (np.array(self.data[pair_idx]['stock2_price'])[1:] -
                             np.array(self.data[pair_idx]['stock2_price'])[:-1]) / \
                            np.array(self.data[pair_idx]['stock2_price'])[:-1]
                        stock2_return = np.insert(stock2_return, 0, 0)
                        observation = np.append(observation, stock1_return[t_idx - 150:t_idx])
                        observation = np.append(observation, stock2_return[t_idx - 150:t_idx])
                elif self.state_length == 300:
                    if self.only_stock_price:
                        # state: only intra day stock price
                        if self.testing:
                            print('input state: only intra day stock price')
                        observation = list()
                        stock1_price = np.asarray(self.data[pair_idx]['stock1_price'][t_idx - 150:t_idx])
                        if np.std(stock1_price) == 0:
                            observation = np.append(observation, (stock1_price - np.mean(stock1_price)) / 1)
                        else:
                            observation = np.append(
                                observation, (stock1_price - np.mean(stock1_price)) / np.std(stock1_price))
                        stock2_price = np.asarray(self.data[pair_idx]['stock2_price'][t_idx - 150:t_idx])
                        if np.std(stock2_price) == 0:
                            observation = np.append(observation, (stock2_price - np.mean(stock2_price)) / 1)
                        else:
                            observation = np.append(
                                observation, (stock2_price - np.mean(stock2_price)) / np.std(stock2_price))
                    elif self.only_stock_price_ln_return:
                        # state: only intra day stock price ln return
                        if self.testing:
                            print('input state: only intra day stock price ln return')
                        observation = list()
                        stock1_return = np.log(np.array(self.data[pair_idx]['stock1_price'])[1:] /
                                               np.array(self.data[pair_idx]['stock1_price'])[:-1])
                        stock1_return = np.insert(stock1_return, 0, 0)
                        stock2_return = np.log(np.array(self.data[pair_idx]['stock2_price'])[1:] /
                                               np.array(self.data[pair_idx]['stock2_price'])[:-1])
                        stock2_return = np.insert(stock2_return, 0, 0)
                        observation = np.append(observation, stock1_return[t_idx - 150:t_idx])
                        observation = np.append(observation, stock2_return[t_idx - 150:t_idx])
                    else:
                        raise ValueError('Unknown input')
                elif self.state_length == 600:
                    if self.testing:
                        print('input state: intra day stock price ln return + intra day stock price')
                    observation = list()
                    stock1_price = np.asarray(self.data[pair_idx]['stock1_price'][t_idx - 150:t_idx])
                    if np.std(stock1_price) == 0:
                        observation = np.append(observation, (stock1_price - np.mean(stock1_price)) / 1)
                    else:
                        observation = np.append(
                            observation, (stock1_price - np.mean(stock1_price)) / np.std(stock1_price))
                    stock2_price = np.asarray(self.data[pair_idx]['stock2_price'][t_idx - 150:t_idx])
                    if np.std(stock2_price) == 0:
                        observation = np.append(observation, (stock2_price - np.mean(stock2_price)) / 1)
                    else:
                        observation = np.append(
                            observation, (stock2_price - np.mean(stock2_price)) / np.std(stock2_price))
                    stock1_return = np.log(np.array(self.data[pair_idx]['stock1_price'])[1:] /
                                           np.array(self.data[pair_idx]['stock1_price'])[:-1])
                    stock1_return = np.insert(stock1_return, 0, 0)
                    stock2_return = np.log(np.array(self.data[pair_idx]['stock2_price'])[1:] /
                                           np.array(self.data[pair_idx]['stock2_price'])[:-1])
                    stock2_return = np.insert(stock2_return, 0, 0)
                    observation = np.append(observation, stock1_return[t_idx - 150:t_idx])
                    observation = np.append(observation, stock2_return[t_idx - 150:t_idx])
                elif self.state_length == 150:
                    # state: spread
                    if self.testing:
                        print(f'{self.pair_idx} input state: only spread')
                    if type(self.data[pair_idx]['mu']) == list:
                        observation = \
                            (np.asarray(self.data[pair_idx]['spread'][t_idx - self.state_length:t_idx]) -
                             np.asarray(self.data[pair_idx]['mu'][t_idx - self.state_length:t_idx])) / \
                            self.data[pair_idx]['stdev']
                    else:
                        observation = \
                            (np.asarray(self.data[pair_idx]['spread'][t_idx - self.state_length:t_idx]) -
                             self.data[pair_idx]['mu']) / self.data[pair_idx]['stdev']
                else:
                    raise ValueError(f'Unknown State Length {self.state_length}')
        else:
            if self.testing:
                print('input state: only no standardization spread')
            if self.state_length == 150:
                observation = np.asarray(self.data[pair_idx]['spread'][t_idx - 150:t_idx])
            else:
                raise NotImplementedError('Not implement without standardization with other state length!')
        assert len(observation) == self.state_length, 'observation length is not match!'
        if self.model_name == 'resnet':
            observation = observation.reshape((int((observation.shape[0]/150)), 150))
        return observation

    def get_known_pt_reward(self):
        action_name = str(self.action[0]) + ',' + str(self.action[1])
        # self.cost_gate = self.action[1]
        if self.date is None:
            self.date = self.data[self.pair_idx]['date']
        if self.date != self.data[self.pair_idx]['date'] and self.testing:
            self.daily_pnl_reset()
        times = self.action[0]
        condition1 = False
        if self.specific_stock is not None:
            if type(self.specific_stock) == np.ndarray:
                check_pair_1 = \
                    self.data[self.pair_idx]['stock1'] == self.specific_stock[0] \
                    and self.data[self.pair_idx]['stock2'] == self.specific_stock[1]
                check_pair_2 = \
                    self.data[self.pair_idx]['stock1'] == self.specific_stock[1] \
                    and self.data[self.pair_idx]['stock2'] == self.specific_stock[0]
                condition1 = not (check_pair_1 or check_pair_2)
            else:
                condition1 = self.data[self.pair_idx]['stock1'] != self.specific_stock \
                             and self.data[self.pair_idx]['stock2'] != self.specific_stock
        if self.data[self.pair_idx]['stdev'] * times <= self.cost_gate or condition1:
            self.less_threshold_count += 1
            return 0, -999
        performance_dict = self.data[self.pair_idx][action_name]
        # print(self.pair_idx)
        record = performance_dict['record']
        wt = performance_dict['reward']
        close_timing = performance_dict['close_timing']
        self.pnl += performance_dict['profit']
        self.pnl_half_tax += performance_dict['half_tax_profit']
        self.pnl_zero_tax += performance_dict['zero_tax_profit']
        self.daily_pnl += performance_dict['profit']
        self.daily_pnl_half_tax += performance_dict['half_tax_profit']
        self.daily_pnl_zero_tax += performance_dict['zero_tax_profit']
        # noinspection PyBroadException
        try:
            self.daily_capital += performance_dict['capital']
        except:
            self.daily_capital += 0
        if performance_dict['capital'] != 0:
            self.pair_half_tax_return_list.append(performance_dict['half_tax_profit']/performance_dict['capital'])
            self.pair_return_list.append(performance_dict['profit']/performance_dict['capital'])
            self.pair_zero_tax_return_list.append(performance_dict['zero_tax_profit']/performance_dict['capital'])
        if record != 0:
            if performance_dict['profit'] > 0:
                self.win_times += 1
            if performance_dict['half_tax_profit'] > 0:
                self.half_cost_win_times += 1
            if performance_dict['zero_tax_profit'] > 0:
                self.zero_cost_win_times += 1
            if self.testing:
                self.total_action.append(self.action)
        if record == 666:
            # Normal close
            wt = 1000 * abs(wt)
            self.total_open_times += 1
            self.normal_close_times += 1
        elif record == -2:
            # stop-loss
            wt = -1000 * abs(wt)
            self.total_open_times += 1
            self.stop_loss_times += 1
        elif record == -4:
            # exit
            wt = -500 * abs(wt)
            self.total_open_times += 1
            self.force_exit_close_times += 1
        elif record == 0:
            # 未開倉
            wt = 0
        else:
            raise ValueError('Unknown status:' + str(record))
        if not self.testing:
            self.training_log.append([self.data[self.pair_idx]['date'], self.data[self.pair_idx]['stock1'],
                                      self.data[self.pair_idx]['stock2'], self.action, performance_dict['record'],
                                      wt, performance_dict['half_tax_profit']])
        else:
            self.testing_log.append([self.data[self.pair_idx]['date'], self.data[self.pair_idx]['stock1'],
                                     self.data[self.pair_idx]['stock2'], self.action, performance_dict['record'],
                                     performance_dict['half_tax_profit'], performance_dict['capital']])
        return wt, close_timing

    def plot_spread(self):
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(30, 15))
        pair_idx = self.pair_idx
        action_name = str(self.action[0]) + ',' + str(self.action[1])
        performance_dict = self.data[pair_idx][action_name]
        std = self.data[pair_idx]['stdev']
        mu = self.data[pair_idx]['mu']
        if type(mu) == float:
            mu = np.array([mu] * len(self.data[pair_idx]['spread']))
        else:
            mu = np.array(mu)
        up_open = mu + self.action[0] * std
        down_open = mu - self.action[0] * std
        up_stop_loss = mu + self.action[1] * std
        down_stop_loss = mu - self.action[1] * std
        plt.plot(self.data[pair_idx]['spread'])
        plt.plot(up_open, 'b')
        plt.plot(down_open, 'b')
        plt.plot(up_stop_loss, 'r')
        plt.plot(down_stop_loss, 'r')
        plt.plot(mu, 'g')
        offset = 149
        plt.vlines(offset, max(up_stop_loss), min(down_stop_loss), 'b', linestyles='dashed')
        close_timing = performance_dict['close_timing']
        record = performance_dict['record']
        if record == 666:
            record_text = 'Normal close'
        elif record == -2:
            record_text = 'Stop loss'
        elif record == -4:
            record_text = 'Force exit'
        elif record == 0:
            record_text = 'No open'
        else:
            raise ValueError('Unknown status:' + str(record))
        if close_timing != -999:
            # Because json data does not contain the timing of opening, we only plot the timing of closing here.
            # If want to see more detailed information, use dump_json_data.py to plot the specific figure instead.
            plt.scatter(close_timing+offset, self.data[pair_idx]['spread'][close_timing+offset],
                        edgecolors='r', marker='o', linewidth=3, zorder=99, facecolors='none')
        file_name = \
            self.data[pair_idx]['date'] + '_' + self.data[pair_idx]['stock1'] + '_' + self.data[pair_idx]['stock2'] \
            + '.jpg'
        if not os.path.exists(self.folder + '/jpg'):
            os.makedirs(self.folder + '/jpg')
        reward_label = f"Profit: {performance_dict['half_tax_profit']} {record_text}"
        plt.title(self.data[pair_idx]['date'] + ' ' + self.data[pair_idx]['stock1'] + ' '
                  + self.data[pair_idx]['stock2'] + ' '
                  f'open threshold:{self.action[0]} stop-loss threshold:{self.action[1]}')
        plt.xlabel(reward_label, wrap=True)
        plt.tight_layout()
        plt.savefig(self.folder + '/jpg/' + file_name)
        plt.close('all')

    def plot_train_rewards(self, title='train_rewards'):
        plotter.plot_train_rewards(self.folder, self.rewards, title=title)
