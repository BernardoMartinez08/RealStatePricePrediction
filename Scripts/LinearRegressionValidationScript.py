# Script de Predicción y Evaluación del Modelo Entrenado

# Tarea 1 - Predicción de Precios de Bienes Raíces
# Asignatura: Sistemas Inteligentes 1298 - Q2-2024
# Docente: Ing. Nicole Rodriguez
# Grupo: OSI TEAM

import sys
import pandas as pandas_reader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib as load_module
import matplotlib.pyplot as plt


def load_data(file_path):
    print("\n\nLoading data from", file_path)
    data = pandas_reader.read_csv(file_path)
    return data


def load_model(model_path):
    print("Loading model from", model_path)
    model = load_module.load(model_path)
    return model


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
    return features, labels, data_without_outliers


def split_data(features, labels):
    print("Splitting data into training and validation sets...")
    features_train, features_validation, labels_train, labels_validation = train_test_split(features, labels, test_size=0.2, random_state=101)
    return features_train, features_validation, labels_train, labels_validation


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


def make_predictions_unseen_data(model, data, num_examples=5):
    print("\n\nPredictions on Unseen Data:")
    unseen_data = data.sample(num_examples, random_state=42)
    unseen_features = unseen_data.drop('price', axis=1)
    unseen_labels = unseen_data['price']
    predictions = predict_price(model, unseen_features)

    for idx, (prediction, actual_price) in enumerate(zip(predictions, unseen_labels)):
        print(f"Example {idx + 1}: Predicted Price: {prediction}, Actual Price: {actual_price}")
        plt.scatter(unseen_labels.iloc[idx], predictions[idx], color='blue')
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual vs Predicted Price")
    plt.show()


def main(model_path, data_path):
    # Cargar el modelo entrenado
    model = load_model(model_path)

    # Cargar los datos de validación
    data = load_data(data_path)

    # Pre-procesar los datos

    features, labels, preprocessed_data = pre_process_data(data)

    # Dividir los datos en conjuntos de entrenamiento y validación
    train_features, validation_features, train_labels, validation_labels = split_data(features, labels)

    # Realizar predicciones con el modelo entrenado
    predictions = predict_price(model, validation_features)

    # Evaluar el modelo
    mae, mse, rmse, r2 = evaluate_model(validation_labels, predictions)

    # Imprimir métricas de evaluación
    print("\n\nModel Evaluation Results:")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R^2 Score:", r2)

    # Después de evaluar el modelo
    make_predictions_unseen_data(model, preprocessed_data)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: py ./LinearRegressionValidationScript.py <model path> <dataset path>")
        sys.exit(1)

    model_input_path = sys.argv[1]
    data_input_path = sys.argv[2]
    main(model_input_path, data_input_path)
