import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


# Инициализируем модель
model = LinearRegression()

# Загрузка данных и обучение модели
for filename in os.listdir('train'):
    if filename.endswith('.csv'):
        filepath = os.path.join('train', filename)

        # Чтение данных из CSV
        df = pd.read_csv(filepath)

        #целевая переменная в последней колонке
        features = df.iloc[:, :-1]
        target = df.iloc[:, -1]

        # Обучение модели
        model.fit(features, target)

        print(f'Model trained on {filename}')

        # Сохранение модели
        joblib.dump(model, f'model_{filename[:-4]}.pkl')
        print(f'Model trained on {filename} and saved as model_{filename[:-4]}.pkl')
