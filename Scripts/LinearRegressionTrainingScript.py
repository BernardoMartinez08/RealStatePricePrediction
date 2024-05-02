# Script de Entrenamiento del Modelo de Regresión Lineal

# Tarea 1 - Predicción de Precios de Bienes Raíces
# Asignatura: Sistemas Inteligentes 1298 - Q2-2024
# Docente: Ing. Nicole Rodriguez
# Grupo: OSI TEAM

import sys
import pandas as pandas_reader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib as save_module


def load_data(file_path):
    print("\n\nLoading data from", file_path)
    data = pandas_reader.read_csv(file_path)
    return data


def remove_outliers(data_frame, column, threshold=1.5):
    print("Removing outliers from", column)
    q1 = data_frame[column].quantile(0.25)
    q3 = data_frame[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return data_frame[(data_frame[column] >= lower_bound) & (data_frame[column] <= upper_bound)]


def remove_outliers_from_columns(data_frame):
    data_without_outliers = remove_outliers(data_frame, 'bath', threshold=1.5)
    data_without_outliers = remove_outliers(data_without_outliers, 'bed', threshold=1.5)
    data_without_outliers = remove_outliers(data_without_outliers, 'acre_lot', threshold=1.5)
    data_without_outliers = remove_outliers(data_without_outliers, 'house_size', threshold=1.5)
    data_without_outliers = remove_outliers(data_without_outliers, 'price', threshold=1.5)
    return data_without_outliers


def pre_process_data(data):
    print("Pre-processing data...")
    proceed_data = data[['price', 'bed', 'bath', 'acre_lot', 'house_size']]
    print("Imputing missing values...")
    imputed_data = proceed_data.fillna(proceed_data.mean())
    data_without_outliers = remove_outliers_from_columns(imputed_data)
    features = data_without_outliers.drop('price', axis=1)
    labels = data_without_outliers['price']
    return features, labels



def split_data(features, labels):
    print("Splitting data into training and validation sets...")
    features_train, features_validation, labels_train, labels_validation = train_test_split(features, labels, test_size=0.2, random_state=101)
    return features_train, features_validation, labels_train, labels_validation


def train_model(features_train, labels_train):
    print("Training the model...")
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(features_train, labels_train)
    return linear_regression_model


def save_model(linear_regression_model, output_file_path):
    save_module.dump(linear_regression_model, output_file_path)
    print("Model saved successfully! on", output_file_path)


def main(data_path, model_path):
    # Cargar los datos
    data = load_data(data_path)

    # Pre-procesar los datos
    features, labels = pre_process_data(data)

    # Dividir los datos en conjuntos de entrenamiento y validación
    train_features, validation_features, train_labels, validation_labels = split_data(features, labels)

    # Entrenar el modelo de regresión lineal
    model = train_model(train_features, train_labels)

    # Guardar el modelo entrenado en un archivo
    save_model(model, model_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Use this script as: py ./LinearRegressionTrainingScript.py <dataset_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = 'price_prediction_linear_regression_model.joblib'
    main(input_file, output_file)
