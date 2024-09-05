from fastapi import FastAPI
from pydantic import BaseModel
from models.model import load_model, predict

# Cargar el modelo al iniciar la API
model = load_model('models/random_forest_model.pkl')

# Definir la estructura de los datos de entrada
class InputData(BaseModel):
    age: int
    flight_distance: int
    inflight_wifi_service: int
    departure_arrival_time_convenient: int
    ease_of_online_booking: int
    gate_location: int
    food_and_drink: int
    online_boarding: int
    seat_comfort: int
    inflight_entertainment: int
    on_board_service: int
    leg_room_service: int
    baggage_handling: int
    checkin_service: int
    inflight_service: int
    cleanliness: int
    departure_delay: int
    arrival_delay: int
    gender: int
    customer_type: int
    travel_type: int
    eco: int
    eco_plus: int
    business: int

# Inicializar la API
app = FastAPI()

# Endpoint para verificar que la API está activa
@app.get("/")
def read_root():
    return {"message": "API para la predicción de satisfacción de clientes está activa"}

# Endpoint para recibir los datos y hacer la predicción real
@app.post("/predict/")
def predict_satisfaction(data: InputData):
    # Convertir los datos de entrada en un formato compatible con el modelo
    input_data = [
        data.age, data.flight_distance, data.inflight_wifi_service,
        data.departure_arrival_time_convenient, data.ease_of_online_booking,
        data.gate_location, data.food_and_drink, data.online_boarding,
        data.seat_comfort, data.inflight_entertainment, data.on_board_service,
        data.leg_room_service, data.baggage_handling, data.checkin_service,
        data.inflight_service, data.cleanliness, data.departure_delay,
        data.arrival_delay, data.gender, data.customer_type, data.travel_type,
        data.eco, data.eco_plus, data.business
    ]

    # Hacer la predicción
    satisfaction = predict(model, input_data)

    return {"satisfaction": "Satisfecho" if satisfaction == 1 else "No Satisfecho/Neutral"}
