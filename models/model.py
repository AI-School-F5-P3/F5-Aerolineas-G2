import joblib

def load_model(path='models/random_forest_proo.pkl'):
    """Carga el modelo desde un archivo .pkl utilizando joblib"""
    # Cargar el modelo usando joblib
    model = joblib.load(path)
    
    # Verifica que el modelo cargado tenga el método predict
    if not hasattr(model, 'predict'):
        raise TypeError("El modelo cargado no tiene el método 'predict'. Verifica el archivo del modelo.")
    
    return model




