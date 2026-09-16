"""Метрики для оценки качества прогнозирования."""

import numpy as np
from sklearn.metrics import mean_absolute_error


def mae(y_true, y_pred):
    """MAE в исходной шкале продаж."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return mean_absolute_error(y_true, y_pred)


def nwrmsle(y_true, y_pred, weights=None):
    """
    Взвешенная RMSLE.

    Для скоропортящихся товаров используется вес 1.25,
    для остальных товаров — 1.0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Прогноз не должен быть отрицательным
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)

    errors = (
        np.log1p(y_pred) - np.log1p(y_true)
    ) ** 2

    if weights is None:
        weights = np.ones_like(errors)

    weights = np.asarray(weights)

    return np.sqrt(
        np.average(errors, weights=weights)
    )


def make_perishable_weights(perishable):
    """Формирование весов для скоропортящихся товаров."""
    perishable = np.asarray(perishable)

    return np.where(perishable == 1, 1.25, 1.0)
