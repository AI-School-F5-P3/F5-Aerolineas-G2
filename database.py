from utils import get_db_connection

def save_to_db(data):
    conn = get_db_connection()
    if conn is None:
        print("No se pudo conectar a la base de datos.")
        return

    cur = conn.cursor()
    
    # Crea la tabla si no existe
    cur.execute("""
        CREATE TABLE IF NOT EXISTS satisfaction_data (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255),
            last_name VARCHAR(255),
            age INTEGER,
            flight_distance INTEGER,
            inflight_wifi_service INTEGER,
            departure_arrival_time_convenient INTEGER,
            ease_of_online_booking INTEGER,
            gate_location INTEGER,
            food_and_drink INTEGER,
            online_boarding INTEGER,
            seat_comfort INTEGER,
            inflight_entertainment INTEGER,
            on_board_service INTEGER,
            leg_room_service INTEGER,
            baggage_handling INTEGER,
            checkin_service INTEGER,
            inflight_service INTEGER,
            cleanliness INTEGER,
            departure_delay INTEGER,
            arrival_delay INTEGER,
            gender INTEGER,
            customer_type INTEGER,
            travel_type INTEGER,
            eco INTEGER,
            eco_plus INTEGER,
            business INTEGER,
            satisfaction INTEGER
        );
    """)
    
    # Inserta los datos en la tabla
    cur.execute("""
        INSERT INTO satisfaction_data (name, last_name, age, flight_distance, inflight_wifi_service, departure_arrival_time_convenient, ease_of_online_booking, gate_location, food_and_drink, online_boarding, seat_comfort, inflight_entertainment, on_board_service, leg_room_service, baggage_handling, checkin_service, inflight_service, cleanliness, departure_delay, arrival_delay, gender, customer_type, travel_type, eco, eco_plus, business, satisfaction)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """), (
        data['name'], data['last_name'], data['age'], data['flight_distance'],
        data['inflight_wifi_service'], data['departure_arrival_time_convenient'],
        data['ease_of_online_booking'], data['gate_location'], data['food_and_drink'],
        data['online_boarding'], data['seat_comfort'], data['inflight_entertainment'],
        data['on_board_service'], data['leg_room_service'], data['baggage_handling'],
        data['checkin_service'], data['inflight_service'], data['cleanliness'],
        data['departure_delay'], data['arrival_delay'], data['gender'],
        data['customer_type'], data['travel_type'], data['eco'], data['eco_plus'],
        data['business'], data['satisfaction']
    )
    
    conn.commit()
    cur.close()
    conn.close()
