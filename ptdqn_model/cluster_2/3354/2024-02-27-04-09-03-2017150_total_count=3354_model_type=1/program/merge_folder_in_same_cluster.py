import configparser
import glob
import os
import re
import shutil


config = configparser.ConfigParser()
config.read('config.ini')
TRAIN_MODEL_ROOT_FOLDER = config['folder']['train_model_root_folder']

if __name__ == '__main__':
    folder = f'{TRAIN_MODEL_ROOT_FOLDER}/tmp26'
    for i in [29, 60, 131, 229, 131, 229, 391, 655, 1076, 1661, 2394, 3354, 4627]:
        all_folder = [f for f in glob.glob(f'{folder}/*_{i}_{i}*') if
                      re.search(r'model_type=(4[0-5]|[2-3][0-9]|1[0-9]|[1-9])', f)]
        if len(all_folder) > 0:
            target_folder = re.sub(r'model_type=(4[0-5]|[2-3][0-9]|1[0-9]|[1-9])', '', all_folder[0])
            print(target_folder)
            os.mkdir(target_folder)
            for source in all_folder:
                shutil.move(source, target_folder)
