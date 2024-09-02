import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Añadir el directorio padre al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_data, clean_data

def plot_satisfaction_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='satisfaction', data=df)
    plt.title('Distribución de Satisfacción del Cliente')
    plt.savefig('notebooks/satisfaction_distribution.png')
    plt.close()

def plot_numeric_features(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_columns = numeric_columns.drop('satisfaction')  # Excluimos la variable objetivo
    
    plt.figure(figsize=(15, 10))
    df[numeric_columns].hist(bins=30, figsize=(20, 15))
    plt.tight_layout()
    plt.savefig('notebooks/numeric_features_distribution.png')
    plt.close()

def plot_correlation_heatmap(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_columns].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Mapa de Calor de Correlaciones')
    plt.savefig('notebooks/correlation_heatmap.png')
    plt.close()

def plot_categorical_features(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=col, hue='satisfaction', data=df)
        plt.title(f'Distribución de {col} por Satisfacción')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'notebooks/{col}_distribution.png')
        plt.close()

def run_eda():
    df = load_data()
    if df is not None:
        df = clean_data(df)
        print(df.info())
        print(df.describe())
        
        plot_satisfaction_distribution(df)
        plot_numeric_features(df)
        plot_correlation_heatmap(df)
        plot_categorical_features(df)
        
        print("Análisis exploratorio de datos completado. Revisa las imágenes generadas en la carpeta 'notebooks'.")
    else:
        print("No se pudo realizar el análisis exploratorio debido a problemas con la carga de datos.")

if __name__ == "__main__":
    run_eda()