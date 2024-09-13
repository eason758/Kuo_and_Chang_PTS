from utils import load_data
import numpy as np
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, KMeans
from sklearn.manifold import TSNE
import json
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import silhouette_score
from minisom import MiniSom
import configparser
import os

config = configparser.ConfigParser()
if os.path.isfile('config.ini'):
    config.read('config.ini')
else:
    # This module is called from agent
    config.read('../config.ini')
PTDQN_DATA_FOLDER = config['folder']['data_root_folder']

def data_preprocess(data, standardization):
    spread = list()
    for i in data:
        if standardization:
            if type(i['mu']) == list:
                spread.append((np.array(i['spread'][:150]) - np.array(i['mu'][:150])) / i['stdev'])
            else:
                spread.append((np.array(i['spread'][:150]) - i['mu']) / i['stdev'])
        else:
            spread.append((np.array(i['spread'][:150])))
    spread = np.array(spread)
    return spread


def clustering(test_year, n_components=3, n_train_months=22, n_test_months=24, n_validation_months=2,
               standardization=True, trend_stationary=False, top_n=None, method='smo'):
    json_name = f'{method}_'
    data_list = ['HsinHuaChang_reverse_random_300_new_std_pickle']

    stock1=None
    stock2=None



    _, _, train_data, test_data = load_data(data_list, test_year-1, 12-n_validation_months+1,
                                            n_validation_months+n_test_months, n_train_months,
                                            testing=True, stock1=stock1,
                                            stock2=stock2, given_action=[[1000000, 2500000]])

    train_spread = data_preprocess(train_data, standardization)
    test_spread = data_preprocess(test_data, standardization)
    print(f'{json_name} train data shape: {train_spread.shape}')
    print(f'{json_name} test data shape: {test_spread.shape}')
    if top_n is not None:
        json_name += f'{top_n}_'
    if not standardization:
        json_name += f'{test_year}_no_standardization'
    else:
        json_name += f'{test_year}'
    if trend_stationary:
        json_name += f'_trend'

    bic = list()
    aic = list()
    silhouette_score_list = list()

    model_type_dict = dict()
    if method in ['k_means', 'gmm']:
        if method == 'k_means':
            model = KMeans(n_clusters=n_components, random_state=666, verbose=1, max_iter=1000)
        else:
            model = mixture.GaussianMixture(n_components=n_components, covariance_type='full', random_state=666,
                                            max_iter=1000)
        model.fit(train_spread)
        bic.append(model.bic(train_spread))
        aic.append(model.aic(train_spread))
        silhouette_score_list.append(silhouette_score(train_spread, model.predict(train_spread)))
        print(f'{n_components} clusters:')
        print(f'bic: {bic}')
        print(f'aic: {aic}')
        print(f'silhouette_score: {silhouette_score_list}')
        y = model.predict(train_spread)
        print(np.unique(y, return_counts=True))
        for i in range(len(train_data)):
            key = train_data[i]['date'] + '_' + train_data[i]['stock1'] + '_' + train_data[i]['stock2']
            model_type_dict[key] = int(y[i]) + 1
        y = model.predict(test_spread)
        for i in range(len(test_data)):
            key = test_data[i]['date'] + '_' + test_data[i]['stock1'] + '_' + test_data[i]['stock2']
            model_type_dict[key] = int(y[i]) + 1
    elif method == 'smo':
        # Examples: https://github.com/JustGlowing/minisom/blob/master/examples/Clustering.ipynb
        som_shape = (1, n_components)
        som = MiniSom(som_shape[0], som_shape[1], train_spread.shape[1], sigma=.5, learning_rate=.5,
                      neighborhood_function='gaussian', random_seed=666)
        som.train_batch(train_spread, 500, verbose=True)
        winner_coordinates = np.array([som.winner(x) for x in train_spread]).T
        cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
        for i in range(len(train_data)):
            key = train_data[i]['date'] + '_' + train_data[i]['stock1'] + '_' + train_data[i]['stock2']
            model_type_dict[key] = int(cluster_index[i]) + 1
        winner_coordinates = np.array([som.winner(x) for x in test_spread]).T
        cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
        for i in range(len(test_data)):
            key = test_data[i]['date'] + '_' + test_data[i]['stock1'] + '_' + test_data[i]['stock2']
            model_type_dict[key] = int(cluster_index[i]) + 1
    else:
        raise ValueError("Unknown method! Only support `k_mean`, `gmm`, and `smo` clustering now!")
    with open(f'{PTDQN_DATA_FOLDER}/{json_name}_model_type_{n_components}.json', 'w') as file:
        json.dump(model_type_dict, file)


if __name__ == '__main__':

    '''******************************* Do Your control below *******************************'''

    # cluster_method : cluster method, ex: K-means, GMM...
    # TOP_N : top n frequently occur pairs, if is not None then extract only Top n pairs, else all pairs

    cluster_method = 'gmm'
    TOP_N = None
    n_components = 3


    test_year = 2018
    n_train_months = 22
    n_test_months = 19
    n_validation_months = 2
      # this will train cluster model from 2015~20161031, and predict cluster for 20161101~20181231
      # because testing year starts from 20170101, and validation period is 2 months,
      # training period is 22 months, testing period is 24 months

    '''******************************* Do Your control above *******************************'''

    clustering(test_year, n_components=n_components,
               n_train_months=n_train_months, n_test_months=n_test_months,
               n_validation_months=n_validation_months,
               method=cluster_method, top_n=TOP_N, trend_stationary=False)
