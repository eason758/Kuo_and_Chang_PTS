import numpy as np
import random
import os
import sys
import tensorflow as tf
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
import json
from ptdqn_env import PTDQN
import pandas as pd
import configparser
from dump_all_cluster_result import dump_final_result


def dump_testing_result_csv(df):
    redundant = ''
    if os.path.isfile(f'{FOLDER}/Test/testing_result.csv'):
        redundant = '-new'
    try:
        df.to_csv(f'{FOLDER}/Test/testing_result{redundant}.csv', index=False)
    except:
        raise ValueError(f'Can not create {FOLDER}/Test/testing_result{redundant}.csv')


def main():
    train_pairs_count, test_pairs_count, train_pairs_data, test_pairs_data = \
        load_data(DATA_FOLDER, TEST_YEAR, TEST_MONTH, TEST_MONTHS, TRAIN_MONTHS, stock1=STOCK1, stock2=STOCK2,
                  testing=True, specific_model_type=SPECIFIC_MODEL_TYPE, given_action=GIVEN_ACTION,
                  model_type_json_file=MODEL_TYPE_JSON_FILE)
    if(len(test_pairs_data)) > 0:
        for a in GIVEN_ACTION:
            if str(a[0]) + ',' + str(a[1]) not in test_pairs_data[0].keys():
                raise ValueError(f'{a} action does not in json data!')
    if test_pairs_count == 0:
        testing_log = []
        df = pd.DataFrame(testing_log, columns=['date', 'stock1', 'stock2', 'action', 'record', 'half_tax_profit',
                                                'capital'])
        test_folder = FOLDER + '/Test'
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        dump_testing_result_csv(df)
        return
    steps = train_pairs_count
    test_steps = test_pairs_count
    all_steps = steps * EPOCHS * STEPS_TIMES
    percentage_explore = INFO['PERCENTAGE_EXPLORE']
    explore_steps = int(all_steps * percentage_explore)  # after how many steps should exploration be stabilized
    if INDEX_DATA is not None:
        with open(f'{DATA_ROOT_FOLDER}/{INDEX_DATA}', 'r') as f:
            index_data = json.load(f)
    else:
        index_data = None
    if STOCK_DAILY_RETURN is not None:
        with open(f'{DATA_ROOT_FOLDER}/{STOCK_DAILY_RETURN}', 'r') as f:
            stock_daily_return_data = json.load(f)
    else:
        stock_daily_return_data = None
    if RESNET:
        model_name = 'resnet'
    else:
        model_name = None
    if METHOD == 'ptdqn':
        env = PTDQN(FOLDER, train_pairs_data, test_pairs_data, window_in=WINDOW_LENGTH, limit_data=DATA_SIZE,
                    formation_period=FORMATION_PERIOD, trading_period=TRADING_PERIOD, testing=True,
                    cost_gate=COST_GATE, state_length=STATE_LENGTH, index_data=index_data, given_action=GIVEN_ACTION,
                    standardization=STANDARDIZATION, only_twse=ONLY_TWSE, fixed_action=None,
                    stock_price_return=STOCK_PRICE_RETURN, stock_daily_return_data=stock_daily_return_data,
                    only_stock_price=ONLY_STOCK_PRICE, only_stock_price_ln_return=ONLY_STOCK_PRICE_LN_RETURN,
                    stock_price_ln_return=STOCK_PRICE_LN_RETURN, model_name=model_name)
    else:
        raise ValueError('env not initialize.')
    model = set_model(env)
    memory = SequentialMemory(limit=MEM_SIZE, window_length=WINDOW_LENGTH)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.0, value_min=0, value_test=0,
                                  nb_steps=explore_steps)
    nb_actions = env.action_space.n  # set up number of actions (outputs)
    dqn = DQNAgent(model=model, gamma=GAMMA, nb_actions=nb_actions, memory=memory, batch_size=BATCH_SIZE,
                   nb_steps_warmup=1000, target_model_update=TAR_MOD_UP, policy=policy, delta_clip=DELTA_CLIP,
                   train_interval=1000, enable_dueling_network=enable_dueling_network)
    dqn.compile(Adam(lr=LR, decay=LR_DEC), metrics=['mse'])
    print('loading weight from ' + env.folder + '/' + 'weights.best.hdf5')
    dqn.load_weights(env.folder + '/' + 'weights.best.hdf5')
    test(env, dqn, test_steps)
    info = dict()
    global ACTION_LIST
    # ACTION_LIST = [int(x) for x in ACTION_LIST]
    info['total_action'] = ACTION_LIST
    try:
        with open(FOLDER + '/test_total_action.json', 'w') as f:
            json.dump(info, f)
    except:
        print('can not establish test_total_action.json')
    # plot_performance_with_dd(DAILY_PROFIT_LIST, DAILY_CAPITAL_LIST, open_number_list, DATE_LIST)


def set_model(env):
    json_file = open(env.folder + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    print(loaded_model.summary())
    return loaded_model


def test(env, dqn, test_steps):
    env.testing = True
    print("testing...")
    env.metrics_reset()
    dqn.test(env, nb_episodes=1 * test_steps * STEPS_TIMES, nb_max_episode_steps=150, visualize=False, verbose=0)
    df = pd.DataFrame(env.testing_log, columns=['date', 'stock1', 'stock2', 'action', 'record', 'half_tax_profit',
                                                'capital'])
    dump_testing_result_csv(df)
    print('less threshold count {}'.format(env.less_threshold_count))
    global LESS_COST_GATE_COUNT
    LESS_COST_GATE_COUNT = env.less_threshold_count
    global PERFORMANCE_LIST
    PERFORMANCE_LIST += [env.pnl, env.pnl_half_tax, env.pnl_zero_tax, env.total_open_times,
                         env.normal_close_times, env.stop_loss_times, env.force_exit_close_times, env.win_times,
                         env.half_cost_win_times, env.zero_cost_win_times]
    global DAILY_PROFIT_LIST
    global DATE_LIST
    global DAILY_CAPITAL_LIST
    global ACTION_LIST
    global PAIR_BASED_SHARPE_RATIO
    if TAX_COST == 0.003:
        DATE_LIST += list(np.array(env.date_list)[np.array(env.daily_pnl_list) != 0])
        DAILY_PROFIT_LIST += list(np.array(env.daily_pnl_list)[np.array(env.daily_pnl_list) != 0])
        r = pd.DataFrame(list(np.array(env.pair_return_list)[np.array(env.pair_return_list) != 0]))
        PAIR_BASED_SHARPE_RATIO = r.mean() / r.std()
    elif TAX_COST == 0.0015:
        DATE_LIST += list(np.array(env.date_list)[np.array(env.daily_pnl_half_tax_list) != 0])
        DAILY_PROFIT_LIST += list(np.array(env.daily_pnl_half_tax_list)[np.array(env.daily_pnl_half_tax_list) != 0])
        r = pd.DataFrame(list(np.array(env.pair_half_tax_return_list)[np.array(env.pair_half_tax_return_list) != 0]))
        print(f'len: {len(env.pair_half_tax_return_list)}')
        PAIR_BASED_SHARPE_RATIO = r.mean() / r.std()
    elif TAX_COST == 0:
        DATE_LIST += list(np.array(env.date_list)[np.array(env.daily_pnl_zero_tax_list) != 0])
        DAILY_PROFIT_LIST += list(np.array(env.daily_pnl_zero_tax_list)[np.array(env.daily_pnl_zero_tax_list) != 0])
        r = pd.DataFrame(list(np.array(env.pair_zero_tax_return_list)[np.array(env.pair_zero_tax_return_list) != 0]))
        PAIR_BASED_SHARPE_RATIO = r.mean() / r.std()
    else:
        raise ValueError('Unknown tax cost')
    DAILY_CAPITAL_LIST += env.daily_capital_list
    ACTION_LIST += env.total_action
    # print(DAILY_PROFIT_LIST, DATE_LIST)

config = configparser.ConfigParser()
config.read('../config.ini')
if config['folder']['root_folder'] not in sys.path:
    sys.path.insert(0, config['folder']['root_folder'])
from utils import load_data  # noqa: E402
DATA_ROOT_FOLDER = config['folder']['data_root_folder']
TRAIN_MODEL_ROOT_FOLDER = config['folder']['train_model_root_folder']
SEED = 20
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(SEED)
random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)
CPU = True
CPU_cores = 1
GPU_mem_use = 0.9
os.environ["CUDA_VISIBLE_DEVICES"] = ""
config = tf.ConfigProto(device_count={'GPU': 0},
                        intra_op_parallelism_threads=CPU_cores,
                        inter_op_parallelism_threads=CPU_cores)
set_session(tf.Session(graph=tf.get_default_graph(), config=config))

EPOCHS = 200
TRAIN_MONTHS = 0
FORMATION_PERIOD = 150
TRADING_PERIOD = 100
METHOD = 'ptdqn'

if __name__ == '__main__':



    '''******************************* Do Your control below *******************************'''

    topn = ""

    # this folder should be inside {ROOT}/ptdqn_model/tmp28/...
    n_cluster_models_folder = r"2018\\"

    # This must be as same as dqn_agent.py
    enable_dueling_network = False

    '''******************************* Do Your control above *******************************'''






    testing_result_list=[]
    for f in os.listdir(TRAIN_MODEL_ROOT_FOLDER+f"/tmp28/{n_cluster_models_folder}/"):

        FOLDER = TRAIN_MODEL_ROOT_FOLDER+f"/tmp28/{n_cluster_models_folder}/"+f

        with open(FOLDER + '/agent_info.json') as INFO_FILE:
            INFO = json.load(INFO_FILE)
        COST_GATE = INFO['COST_GATE']
        if 'TAX_COST' in INFO:
            TAX_COST = INFO['TAX_COST']
        else:
            TAX_COST = 0.0015
        # if COST_GATE == 0:
        #     print('The training cost gate is zero.')
        #     COST_GATE = 0.0015
        DATA_FOLDER = INFO['DATA_FOLDER']
        if type(DATA_FOLDER) == list:
            tmp = list()
            for F in DATA_FOLDER:
                if not os.path.exists(f'{DATA_ROOT_FOLDER}/{F}'):
                    print(f'{F} is not exist! Use original data folder which contains all stock pairs instead.')
                    if 'trend' in F:
                        tmp.append('HsinHuaChang_reverse_random_300_trend_stationary_new_std_pickle')
                    else:
                        tmp.append('HsinHuaChang_reverse_random_300_new_std_pickle')
                else:
                    pass
            if len(tmp) != 0:
                DATA_FOLDER = tmp
        else:
            if not os.path.exists(f'{DATA_ROOT_FOLDER}/{DATA_FOLDER}'):
                print(f'{DATA_FOLDER} is not exist! Use original data folder which contains all stock pairs instead.')
                DATA_FOLDER = 'HsinHuaChang_reverse_random_300_new_std_pickle'
        if 'INDEX_DATA' in INFO:
            INDEX_DATA = INFO['INDEX_DATA']
        else:
            INDEX_DATA = None
        if 'STOCK_DAILY_RETURN' in INFO:
            STOCK_DAILY_RETURN = INFO['STOCK_DAILY_RETURN']
        else:
            STOCK_DAILY_RETURN = None
        if 'STOCK_PRICE_RETURN' in INFO:
            STOCK_PRICE_RETURN = INFO['STOCK_PRICE_RETURN']
        else:
            STOCK_PRICE_RETURN = False
        if 'ONLY_STOCK_PRICE' in INFO:
            ONLY_STOCK_PRICE = INFO['ONLY_STOCK_PRICE']
        else:
            ONLY_STOCK_PRICE = False
        if 'ONLY_STOCK_PRICE_LN_RETURN' in INFO:
            ONLY_STOCK_PRICE_LN_RETURN = INFO['ONLY_STOCK_PRICE_LN_RETURN']
        else:
            ONLY_STOCK_PRICE_LN_RETURN = False
        if 'STOCK_PRICE_LN_RETURN' in INFO:
            STOCK_PRICE_LN_RETURN = INFO['STOCK_PRICE_LN_RETURN']
        else:
            STOCK_PRICE_LN_RETURN = False
        if 'RESNET' in INFO:
            RESNET = INFO['RESNET']
        else:
            RESNET = False
        TEST_YEAR = INFO['TEST_YEAR']
        TEST_MONTH = INFO['TEST_MONTH']
        TEST_MONTHS = INFO['TEST_MONTHS']
        TEST_PAIR_COUNT = 0
        LESS_COST_GATE_COUNT = 0

        WINDOW_LENGTH = INFO['WINDOW_LENGTH']
        GAMMA = INFO['GAMMA']
        LR = INFO['LR']
        LR_DEC = INFO['LR_DEC']
        TAR_MOD_UP = INFO['TAR_MOD_UP']
        DELTA_CLIP = INFO['DELTA_CLIP']
        STATE_LENGTH = INFO['STATE_LENGTH']
        STEPS_TIMES = 1
        SHARPE_RATIO = [0]
        DATA_SIZE = INFO['DATA_SIZE']
        TEST_EPOCHS = 1
        NODES = INFO['NODES']
        BATCH_SIZE = INFO['BATCH_SIZE']
        MEM_SIZE = INFO['MEM_SIZE']
        GIVEN_ACTION = INFO['GIVEN_ACTION']
        try:
            # Non-opening action need to cast to int to match the key in data folder
            GIVEN_ACTION[GIVEN_ACTION.index([1000000, 2500000])] = [int(x) for x in
                                                                    GIVEN_ACTION[GIVEN_ACTION.index([1000000, 2500000])]]
        except ValueError:
            print('Non-opening action does not in given action!')
        if 'STANDARDIZATION' in INFO:
            STANDARDIZATION = INFO['STANDARDIZATION']
        else:
            STANDARDIZATION = True
        if 'ONLY_TWSE' in INFO:
            ONLY_TWSE = INFO['ONLY_TWSE']
        else:
            ONLY_TWSE = False
        if 'SPECIFIC_MODEL_TYPE' in INFO:
            SPECIFIC_MODEL_TYPE = INFO['SPECIFIC_MODEL_TYPE']
        else:
            SPECIFIC_MODEL_TYPE = 0
        if 'MODEL_TYPE_JSON_FILE' in INFO:
            MODEL_TYPE_JSON_FILE = INFO['MODEL_TYPE_JSON_FILE']
        else:
            MODEL_TYPE_JSON_FILE = None
        PERFORMANCE_LIST = list()
        DAILY_PROFIT_LIST = list()
        DAILY_CAPITAL_LIST = list()
        ACTION_LIST = list()
        DATE_LIST = list()
        MDD = 0
        MAX_DAILY_CAPITAL = 0
        PAIR_BASED_SHARPE_RATIO = [0]
        STOCK1 = None
        STOCK2 = None
        if 'STOCK1' in INFO:
            STOCK1 = INFO['STOCK1']
        if 'STOCK2' in INFO:
            STOCK2 = INFO['STOCK2']


        if not os.path.isfile(f'{FOLDER}/Test/testing_result.csv'):
            main()
        print(f'COST_GATE:{COST_GATE}')
        result_df = pd.read_csv(f'{FOLDER}/Test/testing_result.csv')
        results_summary = dump_final_result(result_df)
        for k in results_summary:
            results_summary[k] = [results_summary[k]]
        pd.DataFrame(results_summary, index=[0]).to_csv(f'{FOLDER}/Test/testing_result_summary.csv')
        with open(FOLDER + '/test_total_action.json') as action_file:
            test_total_action = json.load(action_file)
        print(np.unique(test_total_action['total_action'], axis=0, return_counts=True))

        testing_result_list.append(pd.read_csv(FOLDER+"/Test/testing_result.csv"))
    agg_testing_result = pd.concat(testing_result_list)

    performance = dump_final_result(agg_testing_result)
    for k in performance:
        performance[k] = [performance[k]]
    results = pd.DataFrame(performance, index=[0])
    results.to_csv(f"{TRAIN_MODEL_ROOT_FOLDER}/tmp28/{n_cluster_models_folder}/performance.csv")