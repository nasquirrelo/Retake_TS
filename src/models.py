"""Определения основных моделей проекта."""

import torch
import torch.nn as nn
from catboost import CatBoostRegressor


def create_catboost_model(**kwargs):
    """
    Создание CatBoostRegressor.

    По умолчанию используются параметры финальной модели.
    При необходимости параметры можно переопределить через kwargs.
    """
    params = {
        "iterations": 600,
        "depth": 7,
        "learning_rate": 0.08,
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": 50,
        "allow_writing_files": False,
        "thread_count": -1,
    }

    params.update(kwargs)

    return CatBoostRegressor(**params)


class SalesMLP(nn.Module):
    """
    Многослойный перцептрон для прогнозирования log1p(unit_sales).

    Числовые признаки подаются напрямую, категориальные признаки
    преобразуются в обучаемые embeddings.
    """

    def __init__(
        self,
        num_numeric,
        category_sizes,
        embedding_dims,
    ):
        super().__init__()

        if len(category_sizes) != len(embedding_dims):
            raise ValueError(
                "category_sizes и embedding_dims должны иметь одинаковую длину"
            )

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories, embedding_dim)
                for num_categories, embedding_dim
                in zip(category_sizes, embedding_dims)
            ]
        )

        total_embedding_dim = sum(embedding_dims)
        input_dim = num_numeric + total_embedding_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, numeric_features, categorical_features):
        embedded = [
            embedding(categorical_features[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]

        x = torch.cat(
            [numeric_features] + embedded,
            dim=1,
        )

        return self.network(x).squeeze(1)


def create_final_mlp(category_sizes):
    """
    Создание MLP с архитектурой финального эксперимента.
    """

    embedding_dims = [6, 32, 5, 4, 4, 5]

    return SalesMLP(
        num_numeric=14,
        category_sizes=category_sizes,
        embedding_dims=embedding_dims,
    )
