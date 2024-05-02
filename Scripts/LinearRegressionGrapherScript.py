# Script para generar gráficas de dispersión de las características del dataset con el precio de venta

# Tarea 1 - Predicción de Precios de Bienes Raíces
# Asignatura: Sistemas Inteligentes 1298 - Q2-2024
# Docente: Ing. Nicole Rodriguez
# Grupo: OSI TEAM

import sys
import pandas as pandas_reader
import matplotlib.pyplot as plt
import seaborn as sns


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
    features = imputed_data.drop('price', axis=1)
    labels = imputed_data['price']
    return features, labels, imputed_data


def generate_scatter_plots(features, data):
    print("Generating scatter plots with prices...")

    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data[feature], y=data['price'], alpha=0.5)
        plt.title(f'{feature.capitalize()} vs Precio de Venta')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Precio de Venta')
        plt.show()


def main(data_path):
    data = load_data(data_path)

    features, labels, pre_processed_data = pre_process_data(data)
    generate_scatter_plots(features, pre_processed_data)

    bpre_processed_data = remove_outliers_from_columns(pre_processed_data)
    generate_scatter_plots(bpre_processed_data.drop('price', axis=1), bpre_processed_data)

    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Use this script as: py ./LinearRegressionGrapherScript.py <dataset_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    main(input_file)


