import joblib
import pandas as pd

# Cargar el modelo
model = joblib.load('models/random_forest_proo.pkl')

# Obtener los nombres de las características del modelo
feature_names_model = model.feature_names_in_

# Características después de limpiar el dataset
feature_names_input = [
    'Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
    'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
    'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Gender_Male',
    'Customer Type_disloyal Customer', 'Type of Travel_Personal Travel',
    'Class_Eco', 'Class_Eco Plus'
]

# Comparar las características del modelo con las características del dataset
missing_in_model = set(feature_names_input) - set(feature_names_model)
missing_in_input = set(feature_names_model) - set(feature_names_input)

print("Características en el dataset pero no en el modelo:")
print(missing_in_model)

print("\nCaracterísticas en el modelo pero no en el dataset:")
print(missing_in_input)
