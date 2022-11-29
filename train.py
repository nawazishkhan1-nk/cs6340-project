from RunModel import *
import RunModel
from utils import *

# Use this python file to train the deep network

# DATA_DIR_PATH = './devset-official'
# # copy questions and Make train/test split directories on given data and create single json files which the model can read directly
# train_eval_split(DATA_DIR_PATH)
# create_data_json(path_dir='./data/train', f_type='train')
# create_data_json(path_dir='./data/eval', f_type='eval')
# run_model(train_file='./data/train.json', predict_file='./data/eval.json', do_train=True, do_predict=True)


# ---------------After mid-point evaluation------- 
# Make Train-Test Split on devset + testset1
DATA_DIR_PATH = './devset-official-and-testset1'
# copy questions and Make train/test split directories on given data and create single json files which the model can read directly
train_eval_split(DATA_DIR_PATH)
create_data_json(path_dir='./data/train', f_type='train')
create_data_json(path_dir='./data/eval', f_type='eval')
print(f'Train Test Split Done, Now Training----')
RunModel.VERBOSITY = True
run_model(train_file='./data/train.json', predict_file='./data/eval.json', do_train=True, do_predict=True)