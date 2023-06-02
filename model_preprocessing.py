import pandas as pd
from sklearn.preprocessing import StandardScaler

# Инициализируем StandardScaler
scaler = StandardScaler()

# Пути к файлам CSV
filepaths = {
    'regression_normal_train': 'train/regression_normal_train.csv',
    'regression_anomaly_train': 'train/regression_anomaly_train.csv',
}

# Предобработка данных
for name, filepath in filepaths.items():
    # Чтение данных из CSV
    #print(filepaths)
    df = pd.read_csv(filepath)

    # удаляем целевую переменная в последней колонке
    features = df.iloc[:, :-1]
    target = df.iloc[:, -1]

    # Масштабирование признаков
    features_scaled = scaler.fit_transform(features)

    # Обновляем данные в словаре
    #filepaths[name] = (features_scaled, target)

    # Обновляем данные в DataFrame
    df.iloc[:, :-1] = features_scaled

    # Сохраняем обновленные данные в CSV файл
    df.to_csv(filepath, index=False)

#    print (df.head())


