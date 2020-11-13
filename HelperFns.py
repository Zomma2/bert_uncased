
###################################################################
##################### IMPORTS Pt.1 ###########################
###################################################################

# Import to list directories 
import os

print(os.getcwd())
print(os.listdir(os.getcwd()))

### import for time and time monitoring 
import time
## Imports for Metrics operations 
import numpy as np 
## Imports for dataframes and .csv managment
import pandas as pd 

# TensorFlow library  Imports
import tensorflow as tf
print("tf version: ", tf.__version__)

# Keras (backended with Tensorflow) Imports
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger

# Sklearn package for machine learining models 
## WILL BE USED TO SPLIT  TRAIN_VAL DATASETS
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Garbage Collectors
import gc
import sys


## Install Transformers (They are not installed by deafult)
print("running installs...")

os.system('pip install transformers')




# Import Transformers to get Tokenizer for bert and bert models

import transformers
# from transformers import TFAutoModel, AutoTokenizer
# from transformers import RobertaTokenizer, TFRobertaModel
from transformers import DistilBertTokenizer, TFDistilBertModel
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors






def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    return np.array(enc_di['input_ids'])


def build_model(transformer, max_len=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def get_train_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "jigsaw_mjy_train_val_openaug_523200.csv"))
    #test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    return train