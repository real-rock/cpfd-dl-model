from sklearn.metrics import r2_score
import numpy as np


def calc_nmse(real, pred):
    mse = np.sum((real - pred) ** 2)
    size = len(real)
    pred_sum = real.sum()
    real_sum = pred.sum()
    nmse = mse * size / (pred_sum * real_sum)
    return nmse

def calc_fb(real, pred):
    pred_mean = pred.mean()
    real_mean = real.mean()
    return (2 * (pred_mean - real_mean) / (pred_mean + real_mean))

def calc_b(real, pred):
    pred_mean = pred.mean()
    real_mean = real.mean()
    tmp_a = pred - pred_mean
    tmp_b = real - real_mean
    tmp_c = np.sum(np.square(real - real_mean))
    return np.sum(tmp_a * tmp_b) / tmp_c

def calc_a_co(real, pred):
    pred_mean = pred.mean()
    real_mean = real.mean()
    b = calc_b(real, pred)
    return (pred_mean - b * real_mean) / real_mean

def calc_r2(real, pred):
    return r2_score(real, pred)

def calc_corrcoef(real, pred):
    return np.corrcoef(real, pred)[0, 1]

def calc_mse(real, pred):
    return np.sum((real - pred)**2)

def calc_rmse(real, pred):
    return np.sqrt(calc_mse(real, pred))
