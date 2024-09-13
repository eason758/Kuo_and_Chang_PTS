import numpy as np
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, PReLU, Input, BatchNormalization, Conv2D, Add, \
    GlobalAveragePooling2D, Permute
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.backend import set_session
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
import logging
import datetime
import json
import glob
from shutil import copyfile
from ptdqn_env import PTDQN
import sys
import shutil
import stat
import configparser



def build_resnet(input_shape, n_feature_maps, nb_classes):
    print('build conv_x')
    x = Input(shape=input_shape)
    p_x = Permute((3, 2, 1))(x)
    conv_x = BatchNormalization()(p_x)
    conv_x = Conv2D(n_feature_maps, (8, 1), padding='same')(conv_x)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    print('build conv_y')
    conv_y = Conv2D(n_feature_maps, (5, 1), padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
    print('build conv_z')
    conv_z = Conv2D(n_feature_maps, (3, 1), padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)
    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = Conv2D(n_feature_maps, (1, 1), padding='same')(x)
        shortcut_y = BatchNormalization()(shortcut_y)
    else:
        shortcut_y = BatchNormalization()(x)
    print('Merging skip connection')
    y = Add()([shortcut_y, conv_z])
    y = Activation('relu')(y)
    print('build conv_x')
    x1 = y
    conv_x = Conv2D(n_feature_maps*2, (8, 1), padding='same')(x1)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    print('build conv_y')
    conv_y = Conv2D(n_feature_maps*2, (5, 1), padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
    print('build conv_z')
    conv_z = Conv2D(n_feature_maps*2, (3, 1), padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)
    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = Conv2D(n_feature_maps*2, (1, 1), padding='same')(x1)
        shortcut_y = BatchNormalization()(shortcut_y)
    else:
        shortcut_y = BatchNormalization()(x1)
    print('Merging skip connection')
    y = Add()([shortcut_y, conv_z])
    y = Activation('relu')(y)
    print('build conv_x')
    x1 = y
    conv_x = Conv2D(n_feature_maps*2, (8, 1), padding='same')(x1)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    print('build conv_y')
    conv_y = Conv2D(n_feature_maps*2, (5, 1), padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
    print('build conv_z')
    conv_z = Conv2D(n_feature_maps*2, (3, 1), padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)
    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = Conv2D(n_feature_maps*2, (1, 1), padding='same')(x1)
        shortcut_y = BatchNormalization()(shortcut_y)
    else:
        shortcut_y = BatchNormalization()(x1)
    print('Merging skip connection')
    y = Add()([shortcut_y, conv_z])
    y = Activation('relu')(y)
    full = GlobalAveragePooling2D()(y)
    out = Dense(nb_classes, activation='linear')(full)
    print('        -- model was built.')
    return x, out


def main():
    write_model_info()
    if METHOD == ptdqn:
        print(GIVEN_ACTION, len(GIVEN_ACTION))
        if RESNET:
            model_name = 'resnet'
        else:
            model_name = None
        env = PTDQN(FOLDER, TRAIN_PAIRS_DATA, VALIDATE_PAIRS_DATA, window_in=WINDOW_LENGTH, limit_data=DATA_SIZE,
                    formation_period=FORMATION_PERIOD, trading_period=TRADING_PERIOD, cost_gate=COST_GATE,
                    index_data=INDEX_STATE_DATA, state_length=STATE_LENGTH, given_action=GIVEN_ACTION,
                    standardization=STANDARDIZATION, only_twse=ONLY_TWSE, stock_price_return=STOCK_PRICE_RETURN,
                    stock_daily_return_data=STOCK_DAILY_RETURN_DATA, only_stock_price=ONLY_STOCK_PRICE,
                    only_stock_price_ln_return=ONLY_STOCK_PRICE_LN_RETURN, stock_price_ln_return=STOCK_PRICE_LN_RETURN,
                    model_name=model_name)
    else:
        raise NotImplementedError('env not initialize.')
    dqn = set_dqn_agent(env)
    if PRE_TRAINED_MODEL_FOLDER:
        logging.info(f'load model from {PRE_TRAINED_MODEL_FOLDER}')
        dqn.load_weights(f'{PRE_TRAINED_MODEL_FOLDER}/weights.best.hdf5')
    train(env, dqn)
    validate()
    delete_weight_file(FOLDER)


def delete_weight_file(folder):
    shutil.rmtree(f'{folder}/validate', onerror=remove_readonly)
    print(folder)


def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def set_model(env, resnet=False):
    if not resnet:
        model = Sequential()
        model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
        model.add(Dense(NODES))
        model.add(PReLU())
        model.add(Dense(NODES * 2))
        model.add(PReLU())
        model.add(Dense(env.action_space.n))
        model.add(Activation('linear'))
    else:
        x, y = build_resnet((1, int(env.observation_space.shape[0]/150), 150), 64, env.action_space.n)
        model = Model(inputs=x, outputs=y)
    logging.info(model.summary())
    model_json = model.to_json()
    with open(env.folder + '/model.json', "w") as json_file:
        json_file.write(model_json)
    return model


def set_dqn_agent(env):
    model = set_model(env, resnet=RESNET)
    memory = SequentialMemory(limit=MEM_SIZE, window_length=WINDOW_LENGTH)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.0, value_min=0, value_test=0,
                                  nb_steps=EXPLORE_STEPS)
    nb_actions = env.action_space.n  # set up number of actions (outputs)
    dqn = DQNAgent(model=model, gamma=GAMMA, nb_actions=nb_actions, memory=memory, batch_size=BATCH_SIZE,
                   nb_steps_warmup=1000, target_model_update=TAR_MOD_UP, policy=policy, delta_clip=DELTA_CLIP,
                   train_interval=1000, enable_double_dqn=False,enable_dueling_network=enable_dueling_network)
    dqn.compile(Adam(lr=LR, decay=LR_DEC), metrics=['mse'])
    return dqn


def train(env, dqn):
    file_path = env.folder + '/validate/epochs/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    check_pointer = ModelCheckpoint(filepath=file_path + 'weights.{epoch:02d}.hdf5', monitor='val_loss',
                                    verbose=1, save_best_only=False, save_weights_only=True, mode='auto',
                                    period=STEPS)
    dqn.fit(env, nb_steps=EPOCHS * STEPS * STEPS_TIMES, nb_max_episode_steps=150, visualize=False, verbose=0,
            callbacks=[check_pointer])
    env.plot_train_rewards()
    print('del train data')
    global TRAIN_PAIRS_DATA
    del TRAIN_PAIRS_DATA


def validate():
    file_path = FOLDER + '/validate/epochs/'
    xx = [int(x.replace('weights.', '').replace('.hdf5', '')) for x in os.listdir(file_path)]
    xx.sort()
    all_validate_file = [f'weights.{str(x).zfill(2)}.hdf5' for x in xx]
    win_rate_list = list()
    i = 0
    open_number_list = list()
    best_win_rate = -1e200
    best_reward = -1e200
    best_file = None
    best_reward_file = None
    total_reward = []
    if len(VALIDATE_PAIRS_DATA) == 0:
        best_file = all_validate_file[-1]
        logging.info(f'no validate data, use {best_file} as best weight')
        path = file_path + '{}'.format(best_file)
        new_path = FOLDER + '/' + 'weights.best.hdf5'
        logging.info('copy file from {} to {}'.format(path, new_path))
        copyfile(path, new_path)
        return
    if METHOD == 'ptdqn':
        if RESNET:
            model_name = 'resnet'
        else:
            model_name = None
        env = PTDQN(FOLDER, [], VALIDATE_PAIRS_DATA, window_in=WINDOW_LENGTH, limit_data=DATA_SIZE,
                    formation_period=FORMATION_PERIOD, trading_period=TRADING_PERIOD, testing=True,
                    cost_gate=COST_GATE, state_length=STATE_LENGTH, index_data=INDEX_STATE_DATA,
                    given_action=GIVEN_ACTION, standardization=STANDARDIZATION, only_twse=ONLY_TWSE,
                    stock_price_return=STOCK_PRICE_RETURN, stock_daily_return_data=STOCK_DAILY_RETURN_DATA,
                    only_stock_price=ONLY_STOCK_PRICE, only_stock_price_ln_return=ONLY_STOCK_PRICE_LN_RETURN,
                    stock_price_ln_return=STOCK_PRICE_LN_RETURN, model_name=model_name)
    else:
        raise ValueError('env not initialize.')
    dqn = set_dqn_agent(env)
    for validate_file in all_validate_file:
        logging.info('{}:{}'.format(i, validate_file))
        dqn.load_weights(env.folder + '/validate/epochs/' + validate_file)
        # logging.info(f'weight {dqn.model.layers[2].get_weights()}')
        # logging.info(f'weight {dqn.model.layers[4].get_weights()}')
        env.testing = True
        logging.info("validating... {}".format(validate_file))
        env.reset()
        env.metrics_reset()
        dqn.test(env, nb_episodes=1 * TEST_STEPS * STEPS_TIMES, nb_max_episode_steps=150, visualize=False, verbose=0)
        if env.total_open_times == 0:
            win_rate, pnl, open_number = 0, env.pnl_zero_tax, 0
        else:
            if TAX_COST == 0:
                win_rate, pnl, open_number = \
                    env.zero_cost_win_times / env.total_open_times, env.pnl_zero_tax, env.total_open_times
            elif TAX_COST == 0.0015:
                win_rate, pnl, open_number = \
                    env.half_cost_win_times / env.total_open_times, env.pnl_half_tax, env.total_open_times
            elif TAX_COST == 0.003:
                win_rate, pnl, open_number = \
                    env.win_times / env.total_open_times, env.pnl, env.total_open_times
            else:
                raise ValueError('TAX COST does not match!')
        reward = env.rewards[-1]
        total_reward.append(reward)
        win_rate_list.append(win_rate)
        open_number_list.append(open_number)
        i += 1
        if win_rate >= best_win_rate:
            best_file = validate_file
            best_win_rate = win_rate
        if reward >= best_reward:
            best_reward_file = validate_file
            best_reward = reward
        logging.info('win rate:{}'.format(win_rate_list))
        logging.info('open number:{}'.format(open_number_list))
        logging.info('reward:{}'.format(total_reward))
        logging.info('best file:{}'.format(best_file))
        logging.info('best reward file:{}'.format(best_reward_file))
        if len(env.total_action) > 0:
            unique_action = np.unique(env.total_action, axis=0)
            logging.info('action len:{}, {}'.format(len(unique_action), unique_action))
    path = file_path + '{}'.format(best_file)
    new_path = FOLDER + '/' + 'weights.best.hdf5'
    logging.info('copy file from {} to {}'.format(path, new_path))
    copyfile(path, new_path)
    path = file_path + '{}'.format(best_file)
    new_path = FOLDER + '/' + 'weights.best_win_rate.hdf5'
    logging.info('copy file from {} to {}'.format(path, new_path))
    copyfile(path, new_path)
    path = file_path + '{}'.format(best_reward_file)
    new_path = FOLDER + '/' + 'weights.best_reward.hdf5'
    logging.info('copy file from {} to {}'.format(path, new_path))
    copyfile(path, new_path)


def get_layer_output(env, all_validate_file):
    model = set_model(env, resnet=RESNET)
    for validate_file in all_validate_file:
        from keras import backend as k
        np.set_printoptions(threshold=np.inf)
        env.testing = True
        model.load_weights(env.folder + '/validate/epochs/' + validate_file)
        inp = model.input
        outputs = [layer.output for layer in model.layers][1:]
        functor = k.function([inp, k.learning_phase()], outputs)
        logging.info(f'loading {validate_file}')
        xx = env.input_s().reshape(1, 1, 1, 150)
        logging.info(f'idx {env.pair_idx}')
        o = functor([xx, 1.])
        for i in range(len(o)):
            logging.info(f'{i} layer')
            if not os.path.exists(env.folder + '/' + validate_file):
                os.makedirs(env.folder + '/' + validate_file)
            np.save(env.folder + '/' + validate_file + '/' + str(i), o[i])


def write_model_info():
    info = dict()
    info['DATE'] = DATE
    info['EPOCHS'] = EPOCHS
    info['TEST_YEAR'] = TEST_YEAR
    info['TEST_MONTH'] = TEST_MONTH
    info['TEST_MONTHS'] = TEST_MONTHS
    info['WINDOW_LENGTH'] = WINDOW_LENGTH
    info['GAMMA'] = GAMMA
    info['LR'] = LR
    info['LR_DEC'] = LR_DEC
    info['DELTA_CLIP'] = DELTA_CLIP
    info['TAR_MOD_UP'] = TAR_MOD_UP
    info['PERCENTAGE_EXPLORE'] = PERCENTAGE_EXPLORE
    info['DATA_SIZE'] = DATA_SIZE
    info['NODES'] = NODES
    info['BATCH_SIZE'] = BATCH_SIZE
    info['MEM_SIZE'] = MEM_SIZE
    info['STATE_LENGTH'] = STATE_LENGTH
    info['GIVEN_ACTION'] = GIVEN_ACTION
    info['DATA_FOLDER'] = DATA_FOLDER
    info['INDEX_DATA'] = INDEX_DATA
    info['STOCK_DAILY_RETURN'] = STOCK_DAILY_RETURN
    info['VALIDATE_MONTH'] = VALIDATE_MONTH
    info['VALIDATE_YEAR'] = VALIDATE_YEAR
    info['TRAIN_MONTHS'] = TRAIN_MONTHS
    info['VALIDATE_MONTHS'] = VALIDATE_MONTHS
    info['STANDARDIZATION'] = STANDARDIZATION
    info['COST_GATE'] = COST_GATE
    info['ONLY_TWSE'] = ONLY_TWSE
    info['TAX_COST'] = TAX_COST
    info['STOCK_PRICE_RETURN'] = STOCK_PRICE_RETURN
    info['STOCK1'] = STOCK1
    info['STOCK2'] = STOCK2
    info['ONLY_STOCK_PRICE'] = ONLY_STOCK_PRICE
    info['ONLY_STOCK_PRICE_LN_RETURN'] = ONLY_STOCK_PRICE_LN_RETURN
    info['STOCK_PRICE_LN_RETURN'] = STOCK_PRICE_LN_RETURN
    info['RESNET'] = RESNET
    info['SPECIFIC_MODEL_TYPE'] = SPECIFIC_MODEL_TYPE
    info['PRE_TRAINED_MODEL_FOLDER'] = PRE_TRAINED_MODEL_FOLDER
    info['SOURCE_TRAINED_MODEL_FOLDER'] = SOURCE_TRAINED_MODEL_FOLDER
    info['MODEL_TYPE_JSON_FILE'] = MODEL_TYPE_JSON_FILE
    with open(FOLDER + '/agent_info.json', 'w') as file:
        json.dump(info, file)
    program_file = FOLDER + '/program/'
    if not os.path.exists(program_file):
        os.makedirs(program_file)
        for file in sorted(glob.glob('*.py')):
            copyfile(file, program_file + os.path.basename(file))
        for file in sorted(glob.glob('../*.py')):
            copyfile(file, program_file + os.path.basename(file))
        for file in sorted(glob.glob('../dump_json_data/*.py')):
            copyfile(file, program_file + os.path.basename(file))


if __name__ == '__main__':

    '''******************************* Do Your control below *******************************'''

    total_count = None # no Top N

    n_cluster = 3

    TEST_YEAR = 2015
    TEST_MONTH = 1
    TEST_MONTHS = 12
    TRAIN_MONTHS = 22
    VALIDATE_MONTHS = 2

    # This must be as same as trained_dqn_agent.py
    enable_dueling_network = False  # do dueling DQN/DDQN or not

    '''******************************* Do Your control above *******************************'''




    if n_cluster == 1:
        clusters = [0]
    elif n_cluster > 1:
        clusters = range(1,n_cluster+1)
    else:
        raise ValueError("n_cluster must greater or equel to 1")
    for cluster in clusters:

        config = configparser.ConfigParser()
        config.read('../config.ini')
        if config['folder']['root_folder'] not in sys.path:
            sys.path.insert(0, config['folder']['root_folder'])
        from utils import load_data, get_optimal_actions_in_specific_pair  # noqa: E402

        DATA_ROOT_FOLDER = config['folder']['data_root_folder']
        TRAIN_MODEL_ROOT_FOLDER = config['folder']['train_model_root_folder']
        SEED = 20
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(SEED)
        random.seed(SEED)
        tf.compat.v1.set_random_seed(SEED)
        CPU = True
        CPU_cores = 2
        GPU_mem_use = 0.5
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0},
                                          intra_op_parallelism_threads=CPU_cores,
                                          inter_op_parallelism_threads=CPU_cores)
        config.gpu_options.per_process_gpu_memory_fraction = GPU_mem_use
        set_session(tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config))

        EPOCHS = 200

        FORMATION_PERIOD = 150
        TRADING_PERIOD = 100
        ptdqn = 'ptdqn'
        METHOD = ptdqn  # Choose between environments

        # TWSE_DATA = '15_twse_return.json'
        # INDEX_DATA = '20210127_industry_index_return.json'
        ONLY_TWSE = False
        INDEX_DATA = None
        # STOCK_DAILY_RETURN = '20210413_stock_daily_price_return/stock_daily_price_return.json'
        STOCK_DAILY_RETURN = None
        VALIDATE_YEAR = TEST_YEAR - 1
        VALIDATE_MONTH = 11
        STATE_LENGTH = 150
        STOCK_PRICE_RETURN = False
        ONLY_STOCK_PRICE = False
        ONLY_STOCK_PRICE_LN_RETURN = False
        STOCK_PRICE_LN_RETURN = False
        SPECIFIC_MODEL_TYPE = cluster

        STANDARDIZATION = True
        COST_GATE = 0.0015
        TAX_COST = 0.0015
        if TAX_COST == 0:
            TAX_COST = int(TAX_COST)
        #with open("../stock_info/HsinHuaChang_with_trend_TOP_N_STOCK1_IN_2015_2016.txt") as f:
        with open("../stock_info/HsinHuaChang_TOP_N_STOCK1_IN_2015_2016.txt") as f:
            TOP_STOCK1 = [line.strip() for line in f]
        #with open("../stock_info/HsinHuaChang_with_trend_TOP_N_STOCK2_IN_2015_2016.txt") as f:
        with open("../stock_info/HsinHuaChang_with_trend_TOP_N_STOCK2_IN_2015_2016.txt") as f:
            TOP_STOCK2 = [line.strip() for line in f]
        PAIR_NAME = list()
        for a, b in zip(TOP_STOCK1, TOP_STOCK2):
            PAIR_NAME.append(f'{a}_{b}')
        STOCK1 = TOP_STOCK1[:total_count]
        STOCK2 = TOP_STOCK2[:total_count]
        STOCK1 = None
        STOCK2 = None
        DATA_FOLDER = None
        SOURCE_TRAINED_MODEL_FOLDER = None

        if n_cluster == 1:
            MODEL_TYPE_JSON_FILE = None
        else:
            MODEL_TYPE_JSON_FILE = f"gmm_{TEST_YEAR}_model_type_{n_cluster}.json"

        PRE_TRAINED_MODEL_FOLDER = SOURCE_TRAINED_MODEL_FOLDER
        if PRE_TRAINED_MODEL_FOLDER is not None:
            # if str(TEST_YEAR) != PRE_TRAINED_MODEL_FOLDER[20:24]:
            #    raise ValueError('Year not match!')
            if STATE_LENGTH != int(PRE_TRAINED_MODEL_FOLDER[-3:]):
                raise ValueError('State length not match!')
            # if sys.argv[6] != PRE_TRAINED_MODEL_FOLDER[PRE_TRAINED_MODEL_FOLDER.find('method')+6]:
            #    raise ValueError('Representative method not match!')
            PRE_TRAINED_MODEL_FOLDER = f'{TRAIN_MODEL_ROOT_FOLDER}/{PRE_TRAINED_MODEL_FOLDER}'

        FOLDER_DESCRIPTION = f'_{EPOCHS}'
        RESNET = False
        if not STANDARDIZATION:
            FOLDER_DESCRIPTION = FOLDER_DESCRIPTION + 'no_standardization_'
        if ONLY_TWSE:
            FOLDER_DESCRIPTION = FOLDER_DESCRIPTION + 'only_twse_'
        if STOCK_PRICE_RETURN:
            FOLDER_DESCRIPTION = FOLDER_DESCRIPTION + 'stock_price_return_'
        if STOCK_DAILY_RETURN:
            FOLDER_DESCRIPTION = FOLDER_DESCRIPTION + 'stock_daily_return_'
        if ONLY_STOCK_PRICE_LN_RETURN:
            FOLDER_DESCRIPTION = FOLDER_DESCRIPTION + 'only_stock_price_ln_return_'
        if ONLY_STOCK_PRICE:
            FOLDER_DESCRIPTION = FOLDER_DESCRIPTION + 'only_stock_price_'
        if STOCK_PRICE_LN_RETURN:
            FOLDER_DESCRIPTION = FOLDER_DESCRIPTION + 'stock_price_ln_return_'
        if RESNET:
            FOLDER_DESCRIPTION = FOLDER_DESCRIPTION + 'resnet_'
        if SPECIFIC_MODEL_TYPE:
            FOLDER_DESCRIPTION = FOLDER_DESCRIPTION + f'model_type={SPECIFIC_MODEL_TYPE}_'
        if PRE_TRAINED_MODEL_FOLDER:
            FOLDER_DESCRIPTION = FOLDER_DESCRIPTION + 'with_pre_trained_model_'
        if STOCK1 is not None:
            FOLDER_DESCRIPTION = FOLDER_DESCRIPTION + str(STOCK1) + '_' + str(STOCK2) + '_'
        now = datetime.datetime.now()
        DATE = now.strftime('%Y-%m-%d-%H-%M-%S')

        FOLDER = TRAIN_MODEL_ROOT_FOLDER + "/tmp28/" + DATE + "-" + str(TEST_YEAR) + str(
            STATE_LENGTH) + "_total_count=" + str(total_count) + "_model_type=" + str(cluster)

        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                            filename=FOLDER + '/log.txt', level=logging.INFO)

        if DATA_FOLDER is None:
            if SOURCE_TRAINED_MODEL_FOLDER is not None:
                # get the optimal actions from trained model
                with open(f'{TRAIN_MODEL_ROOT_FOLDER}/{SOURCE_TRAINED_MODEL_FOLDER}/agent_info.json', 'r') as f:
                    PRE_TRAINED_MODEL_DATA = json.load(f)
                    DATA_FOLDER = PRE_TRAINED_MODEL_DATA['DATA_FOLDER']
                    GIVEN_ACTION = PRE_TRAINED_MODEL_DATA['GIVEN_ACTION']
                    logging.info(f'Use {DATA_FOLDER} to {SOURCE_TRAINED_MODEL_FOLDER} get given action'
                                 f': {GIVEN_ACTION}')
            else:
                # use training data to get optimal actions
                #SOURCE_DATA_FOLDER = ['HsinHuaChang_reverse_random_300_new_std_pickle',
                #                      'HsinHuaChang_reverse_random_300_trend_stationary_new_std_pickle']
                SOURCE_DATA_FOLDER = ['HsinHuaChang_reverse_random_300_new_std_pickle']
                DATA_FOLDER, GIVEN_ACTION = \
                    get_optimal_actions_in_specific_pair(VALIDATE_YEAR, VALIDATE_MONTH, TRAIN_MONTHS,
                                                         SOURCE_DATA_FOLDER,
                                                         stock1=STOCK1, stock2=STOCK2, testing=False,
                                                         specific_model_type=SPECIFIC_MODEL_TYPE, output_data=False)
                GIVEN_ACTION[GIVEN_ACTION.index([1000000, 2500000])] = [int(x) for x in
                                                                        GIVEN_ACTION[
                                                                            GIVEN_ACTION.index([1000000, 2500000])]]
        else:
            # use exist data folder to load actions
            with open(f'{DATA_ROOT_FOLDER}/{DATA_FOLDER}/agent_action_info.json', 'r') as f:
                GIVEN_ACTION = json.load(f)
        if COST_GATE is None:
            logging.info('Action with cost gate.')
        if INDEX_DATA is not None:
            with open(f'{DATA_ROOT_FOLDER}/{INDEX_DATA}', 'r') as f:
                INDEX_STATE_DATA = json.load(f)
        else:
            INDEX_STATE_DATA = None
        if STOCK_DAILY_RETURN is not None:
            with open(f'{DATA_ROOT_FOLDER}/{STOCK_DAILY_RETURN}', 'r') as f:
                STOCK_DAILY_RETURN_DATA = json.load(f)
        else:
            STOCK_DAILY_RETURN_DATA = None

        TRAIN_PAIRS_COUNT, VALIDATE_PAIRS_COUNT, TRAIN_PAIRS_DATA, VALIDATE_PAIRS_DATA = \
            load_data(DATA_FOLDER, VALIDATE_YEAR, VALIDATE_MONTH, VALIDATE_MONTHS, TRAIN_MONTHS, stock1=STOCK1,
                      stock2=STOCK2,
                      testing=False, specific_model_type=SPECIFIC_MODEL_TYPE, given_action=GIVEN_ACTION,
                      model_type_json_file=MODEL_TYPE_JSON_FILE)
        for a in GIVEN_ACTION:
            if str(a[0]) + ',' + str(a[1]) not in TRAIN_PAIRS_DATA[0].keys():
                raise ValueError(f'{a} action does not in json data!')
        STEPS = TRAIN_PAIRS_COUNT
        TEST_STEPS = VALIDATE_PAIRS_COUNT
        WINDOW_LENGTH = 1
        GAMMA = 0.95
        LR = 1e-5
        LR_DEC = 0
        TAR_MOD_UP = 0.001
        DELTA_CLIP = 1
        STEPS_TIMES = 1
        ALL_STEPS = STEPS * EPOCHS * STEPS_TIMES
        PERCENTAGE_EXPLORE = 0.98
        EXPLORE_STEPS = int(ALL_STEPS * PERCENTAGE_EXPLORE)
        MAX_DATA_SIZE = 12000
        DATA_SIZE = MAX_DATA_SIZE
        TEST_EPOCHS = 1
        NODES = 64
        BATCH_SIZE = 256
        MEM_SIZE = 100000
        main()
