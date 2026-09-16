"""Запуск основного эксперимента CatBoost."""

import numpy as np
import pandas as pd

from config import CATBOOST_PARAMS, FEATURES
from src.metrics import mae, nwrmsle
from src.models import create_catboost_model


def train_catboost(
    train_df,
    val_df,
    target_col="unit_sales",
    weight_col="weight",
):
    """
    Обучение CatBoost и оценка на validation-выборке.

    Ожидается, что train_df и val_df уже содержат
    признаки, сформированные в соответствии с config.py.
    """

    X_train = train_df[FEATURES]
    X_val = val_df[FEATURES]

    # CatBoost обучается на log1p(unit_sales)
    y_train = np.log1p(
        train_df[target_col].clip(lower=0)
    )
    y_val = val_df[target_col].clip(lower=0).to_numpy()

    train_weights = train_df[weight_col].to_numpy()
    val_weights = val_df[weight_col].to_numpy()

    model = create_catboost_model(**CATBOOST_PARAMS)

    model.fit(
        X_train,
        y_train,
        sample_weight=train_weights,
        eval_set=(X_val, np.log1p(y_val)),
        early_stopping_rounds=50,
    )

    pred_log = model.predict(X_val)

    # Возвращение прогноза в исходную шкалу
    predictions = np.expm1(pred_log)
    predictions = np.clip(predictions, 0, None)

    results = {
        "MAE": mae(y_val, predictions),
        "NWRMSLE": nwrmsle(
            y_val,
            predictions,
            weights=val_weights,
        ),
    }

    return model, results


def main():
    """
    Точка входа в эксперимент.

    Подготовка полного датасета Favorita требует значительного
    объёма памяти, поэтому генерация признаков выполняется
    отдельно с помощью функций из src/features.py.
    """
    print("Favorita Grocery Sales Forecasting")
    print("Горизонт прогнозирования: 16 дней")
    print("Основная модель: CatBoost")
    print(
        "Для полного эксперимента сначала необходимо "
        "сформировать train и validation наборы признаков."
    )


if __name__ == "__main__":
    main()
