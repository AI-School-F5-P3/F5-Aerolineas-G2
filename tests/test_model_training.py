import pytest
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.train_model import train_model, split_data

@pytest.fixture
def sample_data():
    # Crear datos de muestra
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y

def test_split_data(sample_data):
    X, y = sample_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

def test_train_model(sample_data):
    X, y = sample_data
    model = train_model(X, y)
    
    assert isinstance(model, RandomForestClassifier)
    assert model.n_features_in_ == 2
    
    # Hacer una predicción para asegurarse de que el modelo funciona
    prediction = model.predict(X.iloc[[0]])
    assert prediction in [0, 1]