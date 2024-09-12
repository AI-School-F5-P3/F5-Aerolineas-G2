import pytest
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.evaluate_model import evaluate_model, evaluate_overfitting

@pytest.fixture
def sample_data_and_model():
    # Crear datos de muestra
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    
    # Crear y entrenar un modelo simple
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model, X, y

def test_evaluate_model(sample_data_and_model):
    model, X, y = sample_data_and_model
    accuracy, report, cm, fpr, tpr, roc_auc = evaluate_model(model, X, y)
    
    assert 0 <= accuracy <= 1
    assert isinstance(report, str)
    assert cm.shape == (2, 2)
    assert len(fpr) == len(tpr)
    assert 0 <= roc_auc <= 1

def test_evaluate_overfitting(sample_data_and_model):
    model, X, y = sample_data_and_model
    train_accuracy, test_accuracy, overfitting = evaluate_overfitting(model, X, y, X, y)
    
    assert 0 <= train_accuracy <= 1
    assert 0 <= test_accuracy <= 1
    assert -1 <= overfitting <= 1