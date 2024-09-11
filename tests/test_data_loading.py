import pytest
import pandas as pd
import os
import sys

# Añadir el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.load_data import load_data

def test_load_data():
    # Crear un DataFrame de prueba
    test_df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    
    # Guardar el DataFrame en un archivo temporal
    temp_file = 'temp_test_data.csv'
    test_df.to_csv(temp_file, index=False)
    
    # Cargar los datos usando la función load_data
    loaded_df = load_data(temp_file)
    
    # Verificar que los datos cargados son correctos
    assert loaded_df.equals(test_df)
    
    # Limpiar: eliminar el archivo temporal
    os.remove(temp_file)

def test_load_data_file_not_found():
    result = load_data('non_existent_file.csv')
    assert result is None