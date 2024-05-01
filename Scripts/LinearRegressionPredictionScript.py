# Script de Predicción y Evaluación del Modelo Entrenado

# Tarea 1 - Predicción de Precios de Bienes Raíces
# Asignatura: Sistemas Inteligentes 1298 - Q2-2024
# Docente: Ing. Nicole Rodriguez
# Grupo: OSI TEAM

import sys
import pandas as pandas_reader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib as load_module


def load_data(file_path):
    print("\n\nLoading data from", file_path)
    data = pandas_reader.read_csv(file_path)
    return data


def load_model(model_path):
    print("Loading model from", model_path)
    model = load_module.load(model_path)
    return model


def pre_process_data(data):
    print("Pre-processing data...")
    proceed_data = data[['price', 'bed', 'bath', 'acre_lot', 'house_size']]
    features = proceed_data.drop('price', axis=1)
    labels = proceed_data['price']
    return features, labels


def predict_price(model, features):
    print("Making predictions...")
    predictions = model.predict(features)
    return predictions


def evaluate_model(labels, predictions):
    print("Evaluating model...")
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(labels, predictions)
    return mae, mse, rmse, r2


def main(model_path, data_path):
    # Cargar el modelo entrenado
    model = load_model(model_path)

    # Cargar los datos de validación
    data = load_data(data_path)

    # Pre-procesar los datos
    features, labels = pre_process_data(data)

    # Realizar predicciones con el modelo entrenado
    predictions = predict_price(model, features)

    # Evaluar el modelo
    mae, mse, rmse, r2 = evaluate_model(labels, predictions)

    # Imprimir métricas de evaluación
    print("Métricas de Evaluación del Modelo:")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R^2 Score:", r2)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: py ./LinearRegressionPredictionScript.py <model path> <dataset path>")
        sys.exit(1)

    model_input_path = sys.argv[1]
    data_input_path = sys.argv[2]
    main(model_input_path, data_input_path)
