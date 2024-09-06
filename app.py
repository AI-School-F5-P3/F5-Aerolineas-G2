import streamlit as st
import requests
import subprocess
import time
from utils import data_access
from database import save_to_db

# URL de la API
API_URL = "http://localhost:8000/predict/"

# Función para iniciar la API
def start_api():
    subprocess.Popen(["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

# Iniciar la API
start_api()

# Esperar un momento para que la API se inicie
time.sleep(5)

# Título de la aplicación
st.title('Predicción de Satisfacción del Cliente de Aerolínea G2')

# Menú de navegación
page = st.sidebar.selectbox("Selecciona una página", ["Formulario", "Datos"])

if page == "Formulario":
    # Formulario para ingresar datos
    st.write("""
        Complete los siguientes campos para predecir si el cliente estará satisfecho o no.
    """)

    # Campos que no se usan para la predicción
    name = st.text_input('Nombre', '')
    last_name = st.text_input('Apellidos', '')

    # Campos para la predicción
    age = st.slider('Edad', 0, 100, 30)
    flight_distance = st.slider('Distancia de vuelo (en km)', 0, 5000, 1000)
    departure_delay = st.slider('Retraso en la salida (minutos)', 0, 240, 15)
    arrival_delay = st.slider('Retraso en la llegada (minutos)', 0, 240, 15)

    # Variables categóricas con botones (0 a 5)
    inflight_wifi_service = st.radio('Inflight wifi service', [0, 1, 2, 3, 4, 5])
    departure_arrival_time_convenient = st.radio('Departure/Arrival time convenient', [0, 1, 2, 3, 4, 5])
    ease_of_online_booking = st.radio('Ease of Online booking', [0, 1, 2, 3, 4, 5])
    gate_location = st.radio('Gate location', [0, 1, 2, 3, 4, 5])
    food_and_drink = st.radio('Food and drink', [0, 1, 2, 3, 4, 5])
    online_boarding = st.radio('Online boarding', [0, 1, 2, 3, 4, 5])
    seat_comfort = st.radio('Seat comfort', [0, 1, 2, 3, 4, 5])
    inflight_entertainment = st.radio('Inflight entertainment', [0, 1, 2, 3, 4, 5])
    on_board_service = st.radio('On-board service', [0, 1, 2, 3, 4, 5])
    leg_room_service = st.radio('Leg room service', [0, 1, 2, 3, 4, 5])
    baggage_handling = st.radio('Baggage handling', [0, 1, 2, 3, 4, 5])
    checkin_service = st.radio('Checkin service', [0, 1, 2, 3, 4, 5])
    inflight_service = st.radio('Inflight service', [0, 1, 2, 3, 4, 5])
    cleanliness = st.radio('Cleanliness', [0, 1, 2, 3, 4, 5])

    gender = st.selectbox('Género', ['Male', 'Female'])
    gender_male = 1 if gender == 'Male' else 0

    customer_type = st.selectbox('Tipo de Cliente', ['Loyal Customer', 'Disloyal Customer'])
    customer_type_disloyal_customer = 1 if customer_type == 'Disloyal Customer' else 0

    travel_type = st.selectbox('Tipo de Viaje', ['Personal Travel', 'Business Travel'])
    type_of_travel_personal_travel = 1 if travel_type == 'Personal Travel' else 0

    travel_class = st.selectbox('Clase de Viaje', ['Eco', 'Eco Plus', 'Business'])
    class_eco = 1 if travel_class == 'Eco' else 0
    class_eco_plus = 1 if travel_class == 'Eco Plus' else 0
    class_business = 1 if travel_class == 'Business' else 0

    # Datos de entrada para la API
    input_data = {
        'age': age,
        'flight_distance': flight_distance,
        'inflight_wifi_service': inflight_wifi_service,
        'departure_arrival_time_convenient': departure_arrival_time_convenient,
        'ease_of_online_booking': ease_of_online_booking,
        'gate_location': gate_location,
        'food_and_drink': food_and_drink,
        'online_boarding': online_boarding,
        'seat_comfort': seat_comfort,
        'inflight_entertainment': inflight_entertainment,
        'on_board_service': on_board_service,
        'leg_room_service': leg_room_service,
        'baggage_handling': baggage_handling,
        'checkin_service': checkin_service,
        'inflight_service': inflight_service,
        'cleanliness': cleanliness,
        'departure_delay': departure_delay,
        'arrival_delay': arrival_delay,
        'gender_male': gender_male,
        'customer_type_disloyal_customer': customer_type_disloyal_customer,
        'type_of_travel_personal_travel': type_of_travel_personal_travel,
        'class_eco': class_eco,
        'class_eco_plus': class_eco_plus,
        'class_business': class_business
    }

    satisfaction_satisfied = None

    # Enviar los datos a la API para obtener la predicción
    if st.button('Predecir Satisfacción'):
        try:
            response = requests.post(API_URL, json=input_data)
            if response.status_code == 200:
                prediction = response.json()
                satisfaction_satisfied = prediction.get('satisfaction')
                st.write(f"Predicción de satisfacción: {satisfaction_satisfied}")
            else:
                st.error(f"Error al conectar con la API. Código de estado: {response.status_code}")
        except requests.RequestException as e:
            st.error(f"Error al conectar con la API: {str(e)}")

    # Enviar los datos a la base de datos
    if st.button('Guardar en la base de datos'):
        db_data = {
            'name': name,
            'last_name': last_name,
            **input_data,
            'satisfaction': satisfaction_satisfied
        }
        success = save_to_db(db_data)
        if success:
            st.success('Datos guardados con éxito')
        else:
            st.error('Error al guardar los datos en la base de datos')

elif page == "Datos":
    data_access()
