import os
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Создание пустого dataframe для сохранения метрик
metrics_df = pd.DataFrame(columns=["Model", "MAE", "MSE", "R2"])

# Загрузка тестовых данных и применение модели
for filename in os.listdir('test'):
    if filename.endswith('.csv'):
        filepath = os.path.join('test', filename)

        # Чтение данных из CSV
        df = pd.read_csv(filepath)

        # Предполагается, что целевая переменная в последней колонке
        features = df.iloc[:, :-1]
        target = df.iloc[:, -1]

        # Удаление последних 4 символов из имени файла и добавление "_model"
        model_name = "model_"+filename[:-8] + "train.pkl"
        print(model_name)
        print(filepath)

        # Загрузка модели
        model = joblib.load(model_name)

        # Получение предсказаний
        predictions = model.predict(features)

        # Вычисление метрик
        mae = mean_absolute_error(target, predictions)
        mse = mean_squared_error(target, predictions)
        r2 = r2_score(target, predictions)

        # Добавление метрик в dataframe
        metrics_df = metrics_df._append({
            "Model": model_name,
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        }, ignore_index=True)

        print(metrics_df)

        # Запись предсказаний в CSV файл
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
        predictions_df.to_csv(f"{filename[:-4]}_predictions.csv", index=False)

        print(f'Predictions for {filename} saved as {filename[:-4]}_predictions.csv')

# Сохранение метрик в CSV файл
metrics_df.to_csv("metrics.csv", index=False)
print('Metrics saved as metrics.csv')