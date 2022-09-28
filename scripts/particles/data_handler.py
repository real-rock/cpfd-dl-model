import pandas as pd
import numpy as np
from datetime import timedelta


def load_data(src=None, method="mean"):
    """
    load_data loads data from src. If there is no file or invalid file in src, it will read from server url.

    Args:
        src: A file path or server url
        method: A group by method(mean or median) to get indoor pm data.
    Returns:
        This returns dataframe which index is datetime format.
    """
    df = None
    try:
        df = pd.read_csv(src, index_col="DATE", parse_dates=True)
    except Exception as e:
        print(e)
        df = pd.read_csv(
            f"http://api.khu-cpfd.com:9019/v1/logs/file",
            index_col="DATE",
            parse_dates=True,
        )
        df.to_csv(src, index_label="DATE")
    return df


def add_pm_diff(df):
    """
    add_pm_diff add PM differences. PM1_2.5, for example, is particle mass from 1 to 2.5 micro gram per m^3.

    Args:
        df: PM dataframe to apply the function.
    Returns:
        This returns new dataframe which contains PM differences.
    """
    cp_df = df.copy()
    cp_df["PM1_2.5"] = cp_df["PM2.5"] - cp_df["PM1"]
    cp_df["PM2.5_10"] = cp_df["PM10"] - cp_df["PM2.5"]
    cp_df["PM1_2.5_OUT"] = cp_df["PM2.5_OUT"] - cp_df["PM1_OUT"]
    cp_df["PM2.5_10_OUT"] = cp_df["PM10_OUT"] - cp_df["PM2.5_OUT"]
    cp_df["PM1_2.5_H_OUT"] = cp_df["PM2.5_H_OUT"] - cp_df["PM1_H_OUT"]
    cp_df["PM2.5_10_H_OUT"] = cp_df["PM10_H_OUT"] - cp_df["PM2.5_H_OUT"]
    return cp_df


def min_max_scale(df, meta_df, **kwargs):
    """
    min_max_scale applies min max scale, which means applied dataframe values will have maximum value of 1 and minimum value of 0.

    Args:
        df: Dataframe to scale.
        meta_df: Dataframe which contains meta data.
        excludes(optional): Columns that will not be scaled. If not it will scale all columns in dataframe.
    Returns:
        This returns new scaled dataframe.
    """
    cols = df.columns
    new_df = df.copy()
    if "excludes" in kwargs.keys():
        cols = [x for x in cols if x not in kwargs["excludes"]]
    for col in cols:
        min_val = meta_df[col]["min"]
        max_val = meta_df[col]["max"]
        new_df[col] = (new_df[col] - min_val) / (max_val - min_val)
    return new_df


def apply_moving_average(df, **kwargs):
    """
    apply_moving_average apply moving average to dataframe.

    Args:
        df: PM dataframe to apply the function.
        window(optional): Window size of moving average. Default is 20.
        min_periods(optional): Minimum number of observations in window required to have a value; otherwise, result is np.nan.
                                min_periods will default to the size of the window.
        center(optional): If False, set the window labels as the right edge of the window index.
                            If True, set the window labels as the center of the window index.
        method(optional): If mean it will average after rolling.
                            If median, it will get median value after rolling; otherwise, it will print error message.
    Returns:
        This returns new dataframe applied moving average.
    """
    window = kwargs_value("window", 20, kwargs)
    min_periods = kwargs_value("min_periods", window, kwargs)
    center = kwargs_value("center", True, kwargs)
    method = kwargs_value("method", "mean", kwargs)

    cols = df.columns
    if "excludes" in kwargs.keys():
        cols = [x for x in df.columns if x not in kwargs["excludes"]]

    if method == "mean":
        return (
            df[cols]
            .resample("1T")
            .mean()
            .rolling(window=window, center=center, min_periods=min_periods)
            .mean()
        )
    elif method == "median":
        return (
            df[cols]
            .resample("1T")
            .mean()
            .rolling(window=window, center=center, min_periods=min_periods)
            .median()
        )
    else:
        print("[ERROR] Invalid moving average method")
        return None


def trim_df(df, dates):
    """
    trim_df extracts dataframes that are in the dates.

    Args:
        df: PM dataframe to apply the function.
        dates: A list of dictionary which keys are `start` and `end`.
    Returns:
        This returns a dataframe list.
    """
    dfs = []
    for date in dates:
        dfs.append(
            df[
                (df.index >= pd.to_datetime(date["start"]))
                & (df.index <= pd.to_datetime(date["end"]))
            ].copy()
        )
    return dfs


def split_dfs(dfs, date):
    """
    split_dfs split dfs from date

    Args:
        dfs: A list of dataframe to be seperated.
        date: A target datetime to seperate dfs.
    Returns:
        This returns two lists of dataframes which seperated from date.
    """
    train_dfs = []
    test_dfs = []
    for df in dfs:
        if df.index[0] < date and df.index[-1] > date:
            train_dfs.append(df.loc[: date - timedelta(minutes=1)])
            test_dfs.append(df.loc[date:])
        elif df.index[0] >= date:
            test_dfs.append(df)
        else:
            train_dfs.append(df)
    return train_dfs, test_dfs


def train_test_split_df(dfs, val_size, test_size):
    """
    train_test_split_df split train, val, test dataframes.

    Args:
        dfs: A list of dataframes to be seperated.
        val_size: Validation dataset size from 0 to 1.
        test_size: Test dataset size from 0 to 1.
    Returns:
        This returns training, validation, test dataframes.
    """
    tot_df = pd.concat(dfs)
    tot_len = len(tot_df)
    train_len = int((1 - val_size - test_size) * tot_len)
    val_len = int(val_size * tot_len)

    train_dfs, test_dfs = split_dfs(dfs, tot_df.index[train_len])
    val_dfs, test_dfs = split_dfs(test_dfs, tot_df.index[train_len + val_len])
    return train_dfs, val_dfs, test_dfs


def kwargs_value(key, defalut, kwargs):
    """
    kwargs_value return value of kwargs if key exists in kwargs. If not it returns default value.

    Args:
        key: A string of key.
        default: Some default value.
        kwargs: A dictionary to be searched.
    Returns:
        This returns a value of kwargs if key exists; otherwise returns default values.
    """
    if key in kwargs.keys():
        return kwargs[key]
    else:
        return defalut


def df_to_dataset(df, inputs, outputs, **kwargs):
    """
    df_to_dataset convert dataframe to numpy dataset.

    Args:
        df: A dataframe source.
        inputs: A list of columns of inputs.
        outputs: A list of columns of outputs.
        in_time_step(optional): Time step in minutes of input sequence. Default is 5.
        out_time_step(optional): Time step in minutes of output sequence. Default is 1.
        offset(optional): Time in minutes between input and output. Default is 1.
                            For example, if offset is 4, output will be 4 minutes after from last time of input.
    Returns:
        This returns dataset X and y.
    """
    in_time_step = kwargs_value("in_time_step", 5, kwargs)
    out_time_step = kwargs_value("out_time_step", 1, kwargs)
    offset = kwargs_value("offset", 1, kwargs)

    X_df = df[inputs]
    y_df = df[outputs]

    frame = in_time_step + out_time_step + offset - 1
    data_size = X_df.shape[0] - frame
    input_size = X_df.shape[1]
    output_size = y_df.shape[1]

    X = np.zeros((data_size, in_time_step, input_size))
    y = np.zeros((data_size, out_time_step, output_size))
    for i in range(data_size):
        x_indices = slice(i, i + in_time_step)
        y_indices = slice(
            i + in_time_step + offset - 1, i + in_time_step + offset - 1 + out_time_step
        )
        X[i] = X_df.values[x_indices]
        y[i] = y_df.values[y_indices]
    return X, y


def dfs_to_dataset(dfs, meta_df, inputs, outputs, **kwargs):
    """
    dfs_to_dataset convert dataframes to numpy dataset.

    Args:
        dfs: A list of dataframes to be converted.
        meta_df: A meta dataframe which will be used to min max scale.
        inputs: A list of columns of inputs.
        outputs: A list of columns of outputs.
        in_time_step(optional): Time step in minutes of input sequence. Default is 5.
        out_time_step(optional): Time step in minutes of output sequence. Default is 1.
        offset(optional): Time in minutes between input and output. Default is 1.
                            For example, if offset is 4, output will be 4 minutes after from last time of input.
        scale(optional): If true, it generates dataset from scaled dataframe; otherwise from original dataframe.
    Returns:
        This returns new DataFrame which contains PM differences.
    """
    Xs, ys = [], []
    scale = kwargs_value("scale", True, kwargs)

    for df in dfs:
        new_df = df.copy()
        if scale:
            new_df = min_max_scale(new_df, meta_df, **kwargs)
        X, y = df_to_dataset(new_df, inputs, outputs, **kwargs)
        Xs.append(X)
        ys.append(y)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    return X, y


def get_datasets(usable_dates, val_size, test_size, **kwargs):
    weather_df = pd.read_csv(
        "../../storage/particle/weather.csv", index_col="DATE", parse_dates=True
    )[["TEMPERATURE", "WIND_DEG", "WIND_SPEED", "HUMIDITY"]]
    weather_df["WIND_DEG"] = np.sin(weather_df["WIND_DEG"].values * np.pi / 180 / 4)

    df_org = load_data()
    df_org = add_pm_diff(df_org)

    excludes = ["PERSON_NUMBER", "AIR_PURIFIER", "AIR_CONDITIONER", "WINDOW", "DOOR"]
    df = apply_moving_average(
        pd.concat([df_org, weather_df], axis=1), min_periods=1, excludes=excludes, **kwargs
    )
    df = pd.concat([df, df_org[excludes]], axis=1)
    df[excludes] = df[excludes].fillna(method="ffill")
    df.dropna(inplace=True)

    dfs = trim_df(df, usable_dates)

    return train_test_split_df(dfs, val_size, test_size)

def get_datasets(inputs, outputs, usable_dates, val_size, test_size, **kwargs):
    weather_df = pd.read_csv(
        "../../storage/particle/weather.csv", index_col="DATE", parse_dates=True
    )[["TEMPERATURE", "WIND_DEG", "WIND_SPEED", "HUMIDITY"]]
    weather_df["WIND_DEG"] = np.sin(weather_df["WIND_DEG"].values * np.pi / 180 / 4)

    df = load_data()
    df = add_pm_diff(df)

    excludes = ["PERSON_NUMBER", "AIR_PURIFIER", "AIR_CONDITIONER", "WINDOW", "DOOR", "WIND_DEG"]
    df_org = pd.concat([df, weather_df], axis=1)
    df = apply_moving_average(
        df_org, min_periods=1, excludes=excludes, **kwargs
    )
    df = pd.concat([df, df_org[excludes]], axis=1)[inputs + outputs]

    dfs = trim_df(df, usable_dates)

    return train_test_split_df(dfs, val_size, test_size)