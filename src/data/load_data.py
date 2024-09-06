import pandas as pd
import os

def load_data(file_path):
    """
    Carga los datos del archivo CSV.
    
    Args:
    file_path (str): Ruta al archivo CSV.
    
    Returns:
    pd.DataFrame: DataFrame con los datos cargados.
    """
    # Obtener la ruta absoluta
    abs_file_path = os.path.abspath(file_path)
    
    if not os.path.exists(abs_file_path):
        print(f"Error: El archivo no existe en la ruta: {abs_file_path}")
        print(f"Directorio actual: {os.getcwd()}")
        print("Contenido del directorio:")
        try:
            print(os.listdir(os.path.dirname(abs_file_path)))
        except FileNotFoundError:
            print("El directorio no existe.")
        return None
    
    try:
        df = pd.read_csv(abs_file_path)
        print(f"Datos cargados exitosamente. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

if __name__ == "__main__":
    # Ejemplo de uso
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(project_root, "data", "raw", "airline_passenger_satisfaction.csv")
    
    print(f"Intentando cargar el archivo desde: {data_path}")
    
    df = load_data(data_path)
    if df is not None:
        print(df.head())
    else:
        print("No se pudieron cargar los datos. Verifica la ruta del archivo y su existencia.")