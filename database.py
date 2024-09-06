from utils import get_db_connection

def save_to_db(data):
    conn = get_db_connection()
    if conn is None:
        print("No se pudo conectar a la base de datos.")
        return False

    cur = conn.cursor()
    
    try:
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
                gender_male INTEGER,
                customer_type_disloyal_customer INTEGER,
                type_of_travel_personal_travel INTEGER,
                class_eco INTEGER,
                class_eco_plus INTEGER,
                class_business INTEGER,
                satisfaction VARCHAR(50)
            );
        """)
        
        # Inserta los datos en la tabla
        cur.execute("""
            INSERT INTO satisfaction_data (name, last_name, age, flight_distance, inflight_wifi_service, 
            departure_arrival_time_convenient, ease_of_online_booking, gate_location, food_and_drink, 
            online_boarding, seat_comfort, inflight_entertainment, on_board_service, leg_room_service, 
            baggage_handling, checkin_service, inflight_service, cleanliness, departure_delay, arrival_delay, 
            gender_male, customer_type_disloyal_customer, type_of_travel_personal_travel, class_eco, class_eco_plus, 
            class_business, satisfaction)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            data['name'], data['last_name'], data['age'], data['flight_distance'],
            data['inflight_wifi_service'], data['departure_arrival_time_convenient'],
            data['ease_of_online_booking'], data['gate_location'], data['food_and_drink'],
            data['online_boarding'], data['seat_comfort'], data['inflight_entertainment'],
            data['on_board_service'], data['leg_room_service'], data['baggage_handling'],
            data['checkin_service'], data['inflight_service'], data['cleanliness'],
            data['departure_delay'], data['arrival_delay'], data['gender_male'],
            data['customer_type_disloyal_customer'], data['type_of_travel_personal_travel'],
            data['class_eco'], data['class_eco_plus'], data['class_business'],
            data['satisfaction']
        ))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error al guardar en la base de datos: {str(e)}")
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()
