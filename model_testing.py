import os
import pandas as pd
import joblib

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

        # Запись предсказаний в CSV файл
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
        predictions_df.to_csv(f"{filename[:-4]}_predictions.csv", index=False)

        print(f'Predictions for {filename} saved as {filename[:-4]}_predictions.csv')
