import joblib
import numpy as np

def load_model(path='models/random_forest_proo.pkl'):
    """Carga el modelo desde un archivo .pkl utilizando joblib"""
    # Cargar el modelo usando joblib
    model = joblib.load(path)
    
    
    # Verifica que el modelo cargado tenga el método predict
    if not hasattr(model, 'predict'):
        raise TypeError("El modelo cargado no tiene el método 'predict'. Verifica el archivo del modelo.")
    
    return model

def predict(model, input_data):
    """Realiza una predicción usando el modelo cargado"""
    # Convierte input_data en un array numpy
    input_array = np.array(input_data).reshape(1, -1)
    
    # Verifica que el modelo tiene el método predict
    if not hasattr(model, 'predict'):
        raise TypeError("El modelo proporcionado no tiene el método 'predict'.")
    
    # Realiza la predicción
    prediction = model.predict(input_array)
    
    return prediction[0]  # Asumiendo que el modelo devuelve una lista/array con una sola predicción



