import logging
import numpy as np
from sklearn import preprocessing
import sys

from utils import load_data  # noqa: E402


def get_input_output(training_action_data, training_stock_price_data, given_action, testing=False):
    if testing:
        output_method = print
    else:
        output_method = logging.info
    output_method(f'get_input_output given_action: {given_action}')
    training_data = list()
    training_label = list()
    for action_data, stock_price_data in zip(training_action_data, training_stock_price_data):
        key1 = f"{action_data['date']}_{action_data['stock1']}_{action_data['stock2']}"
        key2 = f"{stock_price_data['date']}_{stock_price_data['stock1']}_{stock_price_data['stock2']}"
        if key1 != key2:
            raise ValueError(f'Pair does not match! {key1} {key2}')
        tmp = np.zeros((3, 512))
        pair_profit = list()
        for a in given_action:
            pair_profit.append(action_data[f"{a[0]},{a[1]}"]['half_tax_profit'])
        label = np.array(pair_profit).argmax()
        training_label.append(label)
        '''label = np.array(pair_profit).argmax()
        one_hot_label = np.zeros(len(pair_profit))
        one_hot_label[label] = 1
        training_label.append(one_hot_label)'''
        if type(stock_price_data['mu']) == list:
            spread = \
                (np.array(stock_price_data['spread'][:150]) - np.array(stock_price_data['mu'][:150])) / \
                stock_price_data['stdev']
        else:
            spread = \
                (np.array(stock_price_data['spread'][:150]) - stock_price_data['mu']) / stock_price_data['stdev']
        stock1_price = preprocessing.scale(stock_price_data['stock1_price'][16:166])
        stock2_price = preprocessing.scale(stock_price_data['stock2_price'][16:166])
        tmp[0, 181:331] = spread
        tmp[1, 181:331] = stock1_price
        tmp[2, 181:331] = stock2_price
        training_data.append(tmp)
    return np.array(training_data), np.array(training_label)


def data_preprocess(stock_price_data_folder, data_folder, validate_start_year, validate_start_month, validate_months,
                    training_months, given_action, stock1, stock2, specific_model_type=0,
                    model_type_json_file=None, testing=False, second_stage=False):

    # Load data with the performance of each action
    training_action_data_count, validate_action_data_count, training_action_data, validate_action_data = \
        load_data(data_folder, validate_start_year, validate_start_month, validate_months, training_months,
                  stock1=stock1, stock2=stock2, testing=testing, specific_model_type=specific_model_type,
                  given_action=given_action, model_type_json_file=model_type_json_file)

    # Load data with stock price
    training_stock_price_data_count, validate_stock_price_data_count, training_stock_price_data, validate_stock_price_data = \
        load_data(stock_price_data_folder, validate_start_year, validate_start_month,
                                          validate_months, training_months, stock1=stock1, stock2=stock2,
                                          testing=testing, specific_model_type=specific_model_type, given_action=None,
                                          model_type_json_file=model_type_json_file)

    if training_action_data_count != training_stock_price_data_count or \
            validate_action_data_count != validate_stock_price_data_count:
        raise ValueError('Data length does not match!')


    training_data, training_label = get_input_output(training_action_data, training_stock_price_data, given_action)
    validate_data, validate_label = get_input_output(validate_action_data, validate_stock_price_data, given_action)
    if second_stage:
        return training_data, training_label, validate_data, validate_label, training_action_data
    else:
        return training_data, training_label, validate_data, validate_label, validate_action_data
