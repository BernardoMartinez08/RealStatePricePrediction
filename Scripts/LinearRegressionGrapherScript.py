# Script para generar gráficas de dispersión de las características del dataset con el precio de venta

# Tarea 1 - Predicción de Precios de Bienes Raíces
# Asignatura: Sistemas Inteligentes 1298 - Q2-2024
# Docente: Ing. Nicole Rodriguez
# Grupo: OSI TEAM

import sys
import pandas as pandas_reader
import matplotlib.pyplot as plt


def load_data(file_path):
    print("\n\nLoading data from", file_path)
    data = pandas_reader.read_csv(file_path)
    return data


def pre_process_data(data):
    print("Pre-processing data...")
    data = data[['price', 'bed', 'bath', 'acre_lot', 'house_size']]
    features = data.drop('price', axis=1)
    labels = data['price']
    return features, labels


def generate_scatter_plots(features, labels, data):
    print("Generating scatter plots...")
    fig, axs = plt.subplots(nrows=len(features), ncols=1, figsize=(8, 6*len(features)))

    for i, feature in enumerate(features):
        axs[i].scatter(data[feature], labels, alpha=0.5)
        axs[i].set_title(f'{feature.capitalize()} vs Precio de Venta')
        axs[i].set_xlabel(feature.capitalize())
        axs[i].set_ylabel('Precio de Venta')

    plt.tight_layout()
    plt.show()


def main(data_path):
    data = load_data(data_path)
    features, labels = pre_process_data(data)
    generate_scatter_plots(features, labels, data)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Use this script as: py ./LinearRegressionGrapherScript.py <dataset_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    main(input_file)


