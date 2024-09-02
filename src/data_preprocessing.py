import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# Esta función carga datos desde el archivo CSV y los devuelve en un DataFrame de pandas.
def load_data(file_path="data/airline_satisfaction.csv"):
    """
    Carga los datos desde un archivo CSV.
    
    Args:
    file_path (str): Ruta al archivo CSV. Por defecto, "data/airline_satisfaction.csv".
    
    Returns:
    pd.DataFrame: DataFrame con los datos cargados.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Datos cargados exitosamente de {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {file_path}")
        print("Asegúrate de que el archivo esté en la carpeta correcta.")
        return None

def clean_data(df):
    """
    Realiza la limpieza de los datos.
    
    Args:
    df (pd.DataFrame): DataFrame con los datos crudos.
    
    Returns:
    pd.DataFrame: DataFrame con los datos limpios.
    """
    # Eliminar la columna 'Unnamed: 0' si existe
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Eliminar filas duplicadas
    df = df.drop_duplicates()
    
    # Convertir 'satisfaction' a variable binaria
    df['satisfaction'] = df['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
    
    # Manejar valores faltantes
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Para columnas numéricas, rellenar con la mediana
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Para columnas categóricas, rellenar con la moda
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def encode_categorical_variables(df):
    """
    Codifica las variables categóricas.
    
    Args:
    df (pd.DataFrame): DataFrame con los datos limpios.
    
    Returns:
    pd.DataFrame: DataFrame con las variables categóricas codificadas.
    """
    categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    return pd.get_dummies(df, columns=categorical_columns)

def scale_numeric_features(X_train, X_test):
    """
    Escala las características numéricas.
    
    Args:
    X_train (pd.DataFrame): Características de entrenamiento.
    X_test (pd.DataFrame): Características de prueba.
    
    Returns:
    tuple: X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def preprocess_data(df):
    """
    Preprocesa los datos para el entrenamiento del modelo.
    
    Args:
    df (pd.DataFrame): DataFrame con los datos crudos.
    
    Returns:
    tuple: X_train, X_test, y_train, y_test, scaler
    """
    # Limpieza de datos
    df_clean = clean_data(df)
    
    # Codificación de variables categóricas
    df_encoded = encode_categorical_variables(df_clean)
    
    # Separar características y variable objetivo
    X = df_encoded.drop(['id', 'satisfaction'], axis=1)
    y = df_encoded['satisfaction']
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar las características
    X_train_scaled, X_test_scaled, scaler = scale_numeric_features(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print("\nInformación del DataFrame original:")
        print(df.info())
        
        print("\nPrimeras 5 filas del DataFrame original:")
        print(df.head())
        
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        print("\nPreprocesamiento completado.")
        print(f"Forma de X_train: {X_train.shape}")
        print(f"Forma de X_test: {X_test.shape}")
        print(f"Forma de y_train: {y_train.shape}")
        print(f"Forma de y_test: {y_test.shape}")
        
        print("\nColumnas después del preprocesamiento:")
        print(scaler.feature_names_in_)
    else:
        print("No se pudo cargar los datos. Verifica la ubicación del archivo.")