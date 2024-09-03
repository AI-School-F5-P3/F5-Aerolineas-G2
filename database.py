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
            gender INTEGER,
            customer_type INTEGER,
            travel_type INTEGER,
            eco INTEGER,
            eco_plus INTEGER,
            departure_delay INTEGER,
            arrival_delay INTEGER,
            satisfaction INTEGER
        );
    """)
    
    # Inserta los datos en la tabla
    cur.execute("""
        INSERT INTO satisfaction_data (name, last_name, age, gender, customer_type, travel_type, eco, eco_plus, departure_delay, arrival_delay, satisfaction)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """, (data['name'], data['last_name'], data['age'], data['gender'], data['customer_type'], data['travel_type'], data['eco'], data['eco_plus'], data['departure_delay'], data['arrival_delay'], data['satisfaction']))
    
    conn.commit()
    cur.close()
    conn.close()
