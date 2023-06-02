#!/bin/bash

#оператор && между командами гарантирует, что следующая команда будет выполняться только в том случае,
# если предыдущая была выполнена успешно (т.е. вернула код возврата 0).Если какой-либо из Python-скриптов
#завершится с ошибкой, выполнение скрипта прекратится, и последующие команды не будут выполнены.

echo "run pipeline.sh"

# Запуск скрипта предобработки и проверка его успешного завершения
python data_creation.py &&
echo "data_creation.py executed successfully."


# Запуск скрипта обучения и проверка его успешного завершения
python model_preprocessing.py &&
echo "model_preprocessing.py executed successfully."

# Запуск скрипта обучения и проверка его успешного завершения
python model_preparation.py &&
echo "model_preparation.py executed successfully."

# Запуск скрипта тестирования и проверка его успешного завершения
python model_testing.py &&
echo "model_testing.py executed successfully."

echo "All scripts executed successfully."
