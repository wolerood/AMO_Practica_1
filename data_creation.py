import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

# Создаем директории, если они не существуют
if not os.path.exists('train'):
    os.makedirs('train')

if not os.path.exists('test'):
    os.makedirs('test')

# Создаем наборы данных для регрессии
# regression_normal - с небольшим уровнем шума (noice=0.1)
# regression_anomaly - с большим уровнем шума (noice=2.0)
datasets = {
    'regression_normal': make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42),
    'regression_anomaly': make_regression(n_samples=1000, n_features=10, noise=2.0, random_state=42),
}

# Сохраняем наборы данных в CSV-файлах
for name, (X, y) in datasets.items():
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, X.shape[1] + 1)])
    df['target'] = y

    # Разделяем данные на обучение и тест
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # Сохраняем в CSV
    train_df.to_csv(f'train/{name}_train.csv', index=False)
    test_df.to_csv(f'test/{name}_test.csv', index=False)