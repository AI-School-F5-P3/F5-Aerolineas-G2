import pickle

# Cargar el archivo .pkl
with open('models/random_forest_model.pkl', 'rb') as file:
    content = pickle.load(file)

# Imprimir el tipo del objeto cargado y algunos detalles
print(f"Tipo del objeto: {type(content)}")
print(content)