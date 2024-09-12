import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.preprocess_data import preprocess_data

def test_preprocess_data():
    # Crear un DataFrame de prueba
    test_df = pd.DataFrame({
        'Gender': ['Male', 'Female', 'Male'],
        'Customer Type': ['Loyal Customer', 'disloyal Customer', 'Loyal Customer'],
        'Age': [30, 40, 50],
        'Type of Travel': ['Personal Travel', 'Business travel', 'Personal Travel'],
        'Class': ['Eco', 'Business', 'Eco Plus'],
        'satisfaction': ['satisfied', 'neutral or dissatisfied', 'satisfied']
    })

    # Preprocesar los datos
    processed_df = preprocess_data(test_df)

    # Verificar que todas las columnas categóricas se han codificado
    assert processed_df['Gender'].dtype == 'int64'
    assert processed_df['Customer Type'].dtype == 'int64'
    assert processed_df['Type of Travel'].dtype == 'int64'
    assert processed_df['Class'].dtype == 'int64'
    assert processed_df['satisfaction'].dtype == 'int64'

    # Verificar que la columna Age se ha normalizado y su tipo es float64
    assert processed_df['Age'].dtype == 'float64'

    # Verificar que no hay valores nulos
    assert not processed_df.isnull().any().any()