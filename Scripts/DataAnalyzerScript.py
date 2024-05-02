# Script de Análisis de Datos

# Tarea 1 - Predicción de Precios de Bienes Raíces
# Asignatura: Sistemas Inteligentes 1298 - Q2-2024
# Docente: Ing. Nicole Rodriguez
# Grupo: OSI TEAM

import sys
import pandas as pandas_reader
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(file_path):
    print("\n\nLoading data from", file_path)
    data = pandas_reader.read_csv(file_path)
    return data


def pre_process_data(data):
    print("Pre-processing data...")
    proceed_data = data[['price', 'bed', 'bath', 'acre_lot', 'house_size']]
    features = proceed_data.drop('price', axis=1)
    labels = proceed_data['price']
    return features, labels, proceed_data


def data_summary(data, feature):
    stats = data[feature].describe()
    return stats


def data_stats(data, feature):
    max_feature = data[feature].idxmax()
    min_feature = data[feature].idxmin()
    avg_feature = data[feature].mean()
    return max_feature, min_feature, avg_feature


def print_stats(feature, max_feature, min_feature, avg_feature):
    print(f"\nStats of: {feature.capitalize()}")
    print(f"Max value: {max_feature}")
    print(f"Min value: {min_feature}")
    print(f"Avg value: {avg_feature}\n")


def plot_correlation(data):
    corr_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matriz de Correlación entre Características')
    plt.show()


def outlier_analysis(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    threshold = 1.5
    outliers = (data < (q1 - threshold * iqr)) | (data > (q3 + threshold * iqr))
    return outliers


def plot_outliers(data):
    outliers = outlier_analysis(data)
    plt.figure(figsize=(10, 6))
    sns.heatmap(outliers, cmap='viridis', cbar=False)
    plt.title('Valores Atípicos en el Conjunto de Datos')
    plt.xlabel('Características')
    plt.ylabel('Índice de Muestra')
    plt.show()


def plot_distribution(data):
    data.hist(bins=10, figsize=(10, 8))
    plt.suptitle('Distribución de Características', x=0.5, y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()


def remove_outliers(data_frame, column, threshold=1.5):
    print("\n\nRemoving outliers from", column)
    initial_count = len(data_frame)

    q1 = data_frame[column].quantile(0.25)
    q3 = data_frame[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    data_frame = data_frame[(data_frame[column] >= lower_bound) & (data_frame[column] <= upper_bound)]

    final_count = len(data_frame)
    print("Initial count:", initial_count)
    print("Final count:", final_count)

    return data_frame


def remove_outliers_from_columns(data_frame):
    data_without_outliers = remove_outliers(data_frame, 'bath', threshold=1.5)
    data_without_outliers = remove_outliers(data_without_outliers, 'bed', threshold=1.5)
    data_without_outliers = remove_outliers(data_without_outliers, 'acre_lot', threshold=1.5)
    data_without_outliers = remove_outliers(data_without_outliers, 'house_size', threshold=1.5)
    data_without_outliers = remove_outliers(data_without_outliers, 'price', threshold=1.5)
    return data_without_outliers


def main(file_path):
    # Leer los datos
    data = load_data(file_path)
    features, labels, pre_processed_data = pre_process_data(data)

    # Resumen de los datos
    print("\n\nSummary of the data:\n", data.describe())
    print("\n\nNumber of missing values in the data:")
    print("\n\nMissing values in DataSet:\n", data.isnull().sum())
    print("\n\nNumber of non-unique values in the data:\n", data.nunique())
    print("\n\nNumber of non-null values in the data:\n", data.count() - data.isnull().sum())
    print("Duplicate rows in DataSet:", data.duplicated().sum())
    print('\n\nPercentage Missing Value %:\n', data.isna().sum() * 100 / len(data))

    print("\n\nSummary of the features:")
    for feature in features:
        print("\n\nSummary of:", feature.capitalize())
        print(data_summary(data, feature))

    print("\n\nStats of the features:")
    for feature in features:
        max_feature, min_feature, avg_feature = data_stats(data, feature)
        print_stats(feature, max_feature, min_feature, avg_feature)

    print("\n\nShowing Correlation between features:")
    plot_correlation(pre_processed_data)
    print("\n\nShowing Distribution of features:")
    plot_distribution(pre_processed_data)
    print("\n\nShowing Outliers in the data:")
    plot_outliers(pre_processed_data)

    print("\n\nRemoving outliers from the data...")
    data_without_outliers = remove_outliers_from_columns(pre_processed_data)
    print("\n\nStats of the data without outliers:")
    for feature in features:
        print("\n\nSummary of:", feature.capitalize())
        print(data_summary(data_without_outliers, feature))

    print("\n\nShowing Correlation between features without outliers:")
    plot_correlation(data_without_outliers)
    print("\n\nShowing Distribution of features without outliers:")
    plot_distribution(data_without_outliers)
    print("\n\nShowing Outliers in the data without outliers:")
    plot_outliers(data_without_outliers)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Use this script as: py ./DataAnalyzerScript.py <dataset_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    main(input_file)
