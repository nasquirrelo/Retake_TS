# Основные параметры эксперимента

RANDOM_SEED = 42

# Горизонт прогнозирования
FORECAST_HORIZON = 16

# Validation
VALIDATION_START = "2017-07-31"
VALIDATION_END = "2017-08-15"
VALIDATION_HISTORY_END = "2017-07-30"

# Test
TEST_START = "2017-08-16"
TEST_END = "2017-08-31"
TEST_HISTORY_END = "2017-08-15"

# Исторические точки для обучения финальной модели
TRAIN_ORIGINS = [
    "2017-06-19",
    "2017-07-03",
    "2017-07-14",
]

# Признаки финальной модели
FEATURES = [
    "horizon",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "onpromotion",
    "family_code",
    "class",
    "perishable",
    "city_code",
    "state_code",
    "type_code",
    "cluster",
]

# Параметры финальной CatBoost-модели
CATBOOST_PARAMS = {
    "iterations": 600,
    "depth": 7,
    "learning_rate": 0.08,
    "loss_function": "RMSE",
    "random_seed": RANDOM_SEED,
    "verbose": 50,
    "allow_writing_files": False,
    "thread_count": -1,
}

# Параметры MLP
MLP_BATCH_SIZE = 4096
MLP_LEARNING_RATE = 1e-3
MLP_WEIGHT_DECAY = 1e-5
MLP_MAX_EPOCHS = 5
MLP_EARLY_STOPPING_PATIENCE = 2
