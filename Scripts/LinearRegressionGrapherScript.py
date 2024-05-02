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
    proceed_data = data[['price', 'bed', 'bath', 'acre_lot', 'house_size']]
    print("Imputing missing values...")
    imputed_data = proceed_data.dropna()
    features = imputed_data.drop('price', axis=1)
    labels = imputed_data['price']
    return features, labels, imputed_data


def generate_scatter_plots(features, labels, data):
    print("Generating scatter plots...")
    num_features = len(features)
    max_plots_per_figure = 5
    num_figures = (num_features + max_plots_per_figure - 1) // max_plots_per_figure

    for fig_idx in range(num_figures):
        start_idx = fig_idx * max_plots_per_figure
        end_idx = min(start_idx + max_plots_per_figure, num_features)

        fig, axs = plt.subplots(nrows=(end_idx - start_idx), ncols=1, figsize=(8, 6 * (end_idx - start_idx)))

        for i, feature in enumerate(features[start_idx:end_idx]):
            axs[i].scatter(data[feature], labels, alpha=0.5)
            axs[i].set_title(f'{feature.capitalize()} vs Precio de Venta')
            axs[i].set_xlabel(feature.capitalize())
            axs[i].set_ylabel('Precio de Venta')

        plt.tight_layout()
        plt.show()


def main(data_path):
    data = load_data(data_path)
    features, labels, pre_processed_data = pre_process_data(data)
    generate_scatter_plots(features, labels, pre_processed_data)
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Use this script as: py ./LinearRegressionGrapherScript.py <dataset_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    main(input_file)


