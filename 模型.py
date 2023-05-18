from numpy.core.shape_base import atleast_1d
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc, fbeta_score, roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.util.deprecation import HIDDEN_ATTRIBUTE, HiddenTfApiAttribute
import utils.tools as utils
import keras
from keras.engine.topology import Layer
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.constraints import maxnorm
from keras.models import Sequential, Input
from keras.layers import Dense,  Flatten, Activation, Dropout, Embedding
from keras.layers import concatenate,Bidirectional,LSTM, Dropout,TimeDistributed, Permute,Reshape, Lambda, RepeatVector, merge, Input,Multiply
import numpy as np
import tensorflow as tf
from keras.models import Model
import time
from keras.utils import np_utils
import xgboost as xgb
from xgboost import XGBClassifier
import pickle



def model1():
    inputs = Input(shape=(2500, 1,))
    lstm_units = 32
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_initializer='RandomNormal', dropout= 0.3, recurrent_dropout = 0.3,recurrent_initializer='RandomNormal', bias_initializer='zero'))(inputs)     
    attention_mul = attention(lstm_out)
    attention_mul = Flatten()(attention_mul)
    dense_one = Dense(64, kernel_initializer='RandomNormal', bias_initializer='zeros', activation='relu')(attention_mul)
    dense_one = Dropout(0.4)(dense_one)
    dense_two = Dense(32, kernel_initializer='RandomNormal', bias_initializer='zeros', activation='relu')(dense_one)
    output = Dense(2, activation='sigmoid')(dense_two)
    model1 = Model(input= inputs, output=output)
    model1.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model1

def model2():
    model2 = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=6)
    return model2


data_=pd.read_csv(r'file.csv',header=None)
data=np.array(data_)

