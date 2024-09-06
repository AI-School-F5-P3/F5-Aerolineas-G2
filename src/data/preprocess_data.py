import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocesa los datos para el análisis y modelado."""
    
    # Crear una copia del DataFrame para evitar advertencias de SettingWithCopyWarning
    df = df.copy()
    
    # Eliminar columnas innecesarias
    df = df.drop(['Unnamed: 0', 'id'], axis=1, errors='ignore')
    
    # Manejar valores faltantes
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean())
    
    # Codificar variables categóricas
    le = LabelEncoder()
    categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Normalizar variables numéricas
    numerical_cols = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()
    
    return df

if __name__ == "__main__":
    # Definir rutas de archivos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    input_file = os.path.join(project_dir, 'data', 'raw', 'airline_passenger_satisfaction.csv')
    output_file = os.path.join(project_dir, 'data', 'processed', 'cleaned_airlines_data.csv')
    
    # Cargar datos
    print(f"Cargando datos desde {input_file}")
    df_raw = load_data(input_file)
    
    # Preprocesar datos
    print("Preprocesando datos...")
    df_processed = preprocess_data(df_raw)
    
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Guardar datos preprocesados
    df_processed.to_csv(output_file, index=False)
    print(f"Datos preprocesados guardados en {output_file}")