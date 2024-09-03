import streamlit as st
from database import save_to_db

# Título de la aplicación
st.title('Predicción de Satisfacción del Cliente de Aerolínea G2')

# Descripción
st.write("""
    Bienvenido a la aplicación de predicción de satisfacción del cliente.
    Complete los siguientes campos para predecir si el cliente estará satisfecho o no.
""")

# Nombre
name = st.text_input('Nombre', '')

# Apellidos
last_name = st.text_input('Apellidos', '')

# Edad
age = st.slider('Edad', 0, 100, 30)

# Distancia de vuelo
flight_distance = st.slider('Distancia de vuelo (en km)', 0, 5000, 1000)

# Retraso de salida
departure_delay = st.slider('Retraso en la salida (minutos)', 0, 240, 15)

# Retraso de llegada
arrival_delay = st.slider('Retraso en la llegada (minutos)', 0, 240, 15)

# Género
gender = st.selectbox('Género', ['Male', 'Female'])
gender = 1 if gender == 'Male' else 0

# Tipo de Cliente
customer_type = st.selectbox('Tipo de Cliente', ['Loyal Customer', 'Disloyal Customer'])
customer_type = 1 if customer_type == 'Loyal Customer' else 0

# Tipo de Viaje
travel_type = st.selectbox('Tipo de Viaje', ['Personal Travel', 'Business Travel'])
travel_type = 1 if travel_type == 'Personal Travel' else 0

# Clase de Viaje
travel_class = st.selectbox('Clase de Viaje', ['Eco', 'Eco Plus', 'Business'])
eco = 1 if travel_class == 'Eco' else 0
eco_plus = 1 if travel_class == 'Eco Plus' else 0
business = 1 if eco == 0 and eco_plus == 0 else 0

# Simular una predicción
# Aquí deberías conectar con el modelo real que predice la satisfacción
# Por simplicidad, asumiremos que siempre predice "satisfecho"
satisfaction = 1  # 1 para "satisfecho", 0 para "no satisfecho/neutro"

# Mostrar la predicción
st.write(f"Predicción de satisfacción: {'Satisfecho' if satisfaction == 1 else 'No satisfecho/Neutral'}")

# Enviar los datos a la base de datos
if st.button('Guardar en la base de datos'):
    data = {
        'name': name,
        'last_name': last_name,
        'age': age,
        'gender': gender,
        'customer_type': customer_type,
        'travel_type': travel_type,
        'eco': eco,
        'eco_plus': eco_plus,
        'satisfaction': satisfaction
    }
    save_to_db(data)
    st.success('Datos guardados con éxito')
