import pickle

def load_model(path='models/random_forest_model.pkl'):
    """Carga el modelo desde un archivo .pkl"""
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, input_data):
    """Realiza una predicción usando el modelo cargado"""
    # input_data debe ser un array o DataFrame compatible con el modelo
    prediction = model.predict([input_data])
    return prediction[0]  # Asumiendo que el modelo devuelve una lista/array con una sola predicción
