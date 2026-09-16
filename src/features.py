"""Функции для генерации признаков временных рядов."""

import numpy as np
import pandas as pd


LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


def add_calendar_features(df, date_col="date"):
    """Добавление календарных признаков."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_month"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["is_weekend"] = (
        df["day_of_week"].isin([5, 6]).astype("int8")
    )

    return df


def add_horizon(df, origin_date, date_col="date"):
    """Номер шага прогноза относительно forecast origin."""
    df = df.copy()
    origin_date = pd.Timestamp(origin_date)

    df["horizon"] = (
        pd.to_datetime(df[date_col]) - origin_date
    ).dt.days

    return df


def prepare_unit_sales(df):
    """
    Подготовка целевой переменной.

    Отрицательные продажи соответствуют возвратам,
    поэтому для задачи прогнозирования спроса заменяются на 0.
    """
    df = df.copy()
    df["unit_sales"] = df["unit_sales"].clip(lower=0)

    return df


def make_sales_matrix(
    history,
    value_col="unit_sales",
):
    """
    Представление истории продаж в виде матрицы
    store-item x date.
    """
    matrix = history.pivot_table(
        index=["store_nbr", "item_nbr"],
        columns="date",
        values=value_col,
        aggfunc="sum",
        fill_value=0,
    )

    return matrix


def get_lag_feature(
    sales_matrix,
    origin_date,
    lag,
):
    """Получение lag-признака для заданной точки прогноза."""
    origin_date = pd.Timestamp(origin_date)
    lag_date = origin_date - pd.Timedelta(days=lag - 1)

    if lag_date not in sales_matrix.columns:
        return pd.Series(
            0.0,
            index=sales_matrix.index,
        )

    return sales_matrix[lag_date].astype("float32")


def get_rolling_mean(
    sales_matrix,
    origin_date,
    window,
):
    """Среднее значение продаж за предыдущие window дней."""
    origin_date = pd.Timestamp(origin_date)

    start_date = origin_date - pd.Timedelta(days=window - 1)
    dates = pd.date_range(start_date, origin_date)

    available_dates = [
        date for date in dates
        if date in sales_matrix.columns
    ]

    if not available_dates:
        return pd.Series(
            0.0,
            index=sales_matrix.index,
        )

    # Деление на полный размер окна учитывает дни,
    # в которые продажи отсутствовали.
    values = sales_matrix[available_dates].sum(axis=1) / window

    return values.astype("float32")


def add_history_features(
    feature_df,
    sales_matrix,
    origin_date,
):
    """Добавление lag и rolling-признаков."""
    df = feature_df.copy()

    pair_index = pd.MultiIndex.from_frame(
        df[["store_nbr", "item_nbr"]]
    )

    for lag in LAGS:
        lag_values = get_lag_feature(
            sales_matrix,
            origin_date,
            lag,
        )
        df[f"lag_{lag}"] = lag_values.reindex(
            pair_index,
            fill_value=0,
        ).to_numpy()

    for window in ROLLING_WINDOWS:
        rolling_values = get_rolling_mean(
            sales_matrix,
            origin_date,
            window,
        )
        df[f"rolling_mean_{window}"] = rolling_values.reindex(
            pair_index,
            fill_value=0,
        ).to_numpy()

    return df
