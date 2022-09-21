from pyDOE import lhs

import pandas as pd
import tensorflow as tf
import numpy as np

import time
import json
import os

import sys
sys.path.append("../scripts/particles/")

import data_handler as dh
import metrics
import utils

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    SimpleRNN,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    MaxPooling1D,
)
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

outputs = ['PM1', 'PM2.5', 'PM10']
inputs = [
    'PM1_2.5_OUT', 
    'PM1_2.5_H_OUT',
    'PM2.5_OUT', 
    'PM2.5_H_OUT',
    'PM2.5_10_OUT',
    'PM2.5_10_H_OUT',
    'PERSON_NUMBER',
    'AIR_PURIFIER',
    'WINDOW',
    'AIR_CONDITIONER',
    'DOOR',
    # 'TEMPERATURE',
    # 'WIND_SPEED',
    'WIND_DEG',
    'HUMIDITY'
]

dates = [
    {"start": "2022-05-07 09:40", "end": "2022-05-17 08:38"},
    {"start": "2022-05-17 11:25", "end": "2022-05-30 23:26"},
    {"start": "2022-06-01 22:40", "end": "2022-07-02 07:00"},
    {"start": "2022-07-02 16:40", "end": "2022-07-09 07:13"},
    {"start": "2022-07-09 14:30", "end": "2022-07-12 10:00"},
    {"start": "2022-07-25 12:00", "end": "2022-08-01 10:00"},
    {"start": "2022-08-03 09:00", "end": "2022-08-11 22:18"},
    {"start": "2022-08-12 12:14", "end": "2022-08-20 00:00"},
    {"start": "2022-08-20 09:38", "end": "2022-09-01 00:00"},
]

moving_average_window = 20
moving_average_method = 'mean'
val_size = 0.15
test_size = 0.25
train_size = 1 - val_size - test_size

weather_df = pd.read_csv('../../storage/particle/weather.csv', index_col='DATE', parse_dates=True)[['TEMPERATURE', 'WIND_DEG', 'WIND_SPEED', 'HUMIDITY']]
weather_df['WIND_DEG'] = np.sin(weather_df['WIND_DEG'].values * np.pi / 180)

df_org = dh.load_data("../../storage/particle/data.csv")
df_org = dh.add_pm_diff(df_org)

excludes = ['PERSON_NUMBER', 'AIR_PURIFIER', 'AIR_CONDITIONER', 'WINDOW', 'DOOR']
df = dh.apply_moving_average(pd.concat([df_org, weather_df], axis=1), 
                             window=moving_average_window, 
                             method=moving_average_method, 
                             excludes=excludes)
df = pd.concat([df, df_org[excludes]], axis=1)
df[excludes] = df[excludes].fillna(method='ffill')
df.dropna(inplace=True)

dfs = dh.trim_df(df, dates)
train_dfs, val_dfs, test_dfs = dh.train_test_split_df(dfs, val_size, test_size)
meta_df = pd.concat(train_dfs).describe()

lr_val = [0.001, 0.0001, 0.00001]

basic_params = {
    "window_size": [12, 60],
    "pool_size": [2, 6],
    "pool_strides": [1, 4],
    "dense": {
        "units": [32, 256],
        "dropout": [0, 0.5],
        "leaky_relu": [0, 0.5],
    },
    "batch_size": [32, 256],
    "lr": [0, 2],
}

conv_params = {
    "conv": {
        "filters": [32, 256],
        "kernel_size": [3, 7],
        "strides": [0, 3],
    },
}

param_dict = {}

for k in basic_params.keys():
    if type(basic_params[k]) == dict:
        for k2 in basic_params[k]:
            param_dict[k+'_'+k2] = basic_params[k][k2]
    else:
        param_dict[k] = basic_params[k]
        
for k in conv_params.keys():
    if type(conv_params[k]) == dict:
        for k2 in conv_params[k]:
            param_dict[k+'_'+k2] = conv_params[k][k2]
    else:
        param_dict[k] = conv_params[k]
        
root_dir = '../../projects/particle/lhs_opt'
proj_dir = '../../projects/particle/lhs_opt/2022-09-10_11:43'
conv_idc = np.load(f'{proj_dir}/conv_idc.npy')

def li_to_dt(li):
    dt = {}
    for i, k in enumerate(param_dict.keys()):
        if k != 'dense_dropout' and k != 'dense_leaky_relu':
            dt[k] = int(li[i])
        else:
            dt[k] = li[i]
    return dt


def model_builder(p_dt):
    input_tensor = Input(shape=(p_dt["window_size"], len(inputs)), name="input")
    x = input_tensor
    if p_dt["conv_strides"] == 0:
        p_dt["conv_strides"] = None
    x = Conv1D(p_dt["conv_filters"], 
               kernel_size=p_dt["conv_kernel_size"], 
               kernel_initializer='he_uniform', 
               activation='relu', 
               strides=p_dt["conv_strides"],
               padding='same')(x)
    x = MaxPooling1D(pool_size=p_dt["pool_size"], 
                     strides=p_dt["pool_strides"], 
                     padding='same')(x)
    x = Flatten()(x)
    x = Dense(p_dt["dense_units"], 
              kernel_initializer='he_uniform', 
              activation=LeakyReLU(p_dt["dense_leaky_relu"]))(x)
    x = Dropout(p_dt["dense_dropout"])(x)
    output = Dense(len(outputs), kernel_initializer='he_uniform', activation="relu", name="output")(x)

    _model = Model(
        inputs=input_tensor,
        outputs=output,
        name='test',
    )

    _model.compile(
        optimizer=Adam(learning_rate=lr_val[p_dt["lr"]]),
        loss='mse',
        metrics=RootMeanSquaredError(),
    )
    return _model


rlr_cb = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, mode="min", min_lr=1e-6, verbose=False
)
ely_cb = EarlyStopping(monitor="val_loss", patience=15, mode="min", verbose=False, restore_best_weights=True)

def calc_metric(real, pred):
    metric_funcs = [metrics.calc_r2,
                    metrics.calc_corrcoef, 
                    metrics.calc_nmse, 
                    metrics.calc_fb,
                    metrics.calc_b,
                    metrics.calc_a_co, 
                    metrics.calc_mse,]
    res = np.zeros(len(metric_funcs))

    for i, metric in enumerate(metric_funcs):
        res[i] = metric(real, pred)
    return res


def to_dataset(_dfs, in_time_step):
    return dh.dfs_to_dataset(_dfs, meta_df, inputs, outputs, in_time_step=in_time_step, out_time_step=1, offset=1, excludes=outputs)

def train_model(_model):
    _ = model.fit(
        x=X_train,
        y=y_train,
        batch_size=p["batch_size"],
        shuffle=False,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=[rlr_cb, ely_cb],
        verbose=False,
    )
    print(f'[INFO] Finished training')
    K.clear_session()

idc = conv_idc
metrics_indices = ['r2', 'corr', 'nmse', 'fb', 'b', 'a/c', 'mse']

metric_df = pd.DataFrame(np.zeros((len(conv_idc), len(metrics_indices))), columns=metrics_indices)
if os.path.exists(f'{proj_dir}/metric.csv'):
    print(f'Found metric_df. Read from source.')
    metric_df = pd.read_csv(f'{proj_dir}/metric.csv', index_col='index')


for i, conv_idx in enumerate(idc):
    root_dir = proj_dir+f'/trial{i:03d}'
    if os.path.exists(root_dir):
        continue
    os.makedirs(root_dir)
    print(f'[INFO] Trial{i:03d} training start')

    p = li_to_dt(conv_idx)
    with open(f"{root_dir}/params.json", "w") as outfile:
        json.dump(p, outfile)
        outfile.close()

    win_size = p["window_size"]
    X_train, y_train = to_dataset(train_dfs, win_size)
    X_val, y_val = to_dataset(val_dfs, win_size)
    X_test, y_test = to_dataset(test_dfs, win_size)

    y_train = y_train.reshape(-1, len(outputs))
    y_val = y_val.reshape(-1, len(outputs))
    y_test = y_test.reshape(-1, len(outputs))
    model = model_builder(p)
    train_model(model)

    y_hat = model.predict(X_test, verbose=False)
    print(f'[INFO] Trial{i:03d} finished predict')
    # tf.compat.v1.reset_default_graph()
    # del model

    metric = calc_metric(y_test, y_hat)
    metric_df.iloc[i] = metric
    print(f'[INFO] Trial{i:03d} successfully ended.. Saving metrics')
    metric_df.to_csv(f'{proj_dir}/metric.csv', index_label='index')
    K.clear_session()
