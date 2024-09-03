import psycopg2
from psycopg2 import sql

# Función para conectar a la base de datos PostgreSQL
def connect_db():
    conn = psycopg2.connect(
        dbname='aerolinea',        # Reemplaza con el nombre de tu base de datos
        user='postgres',              # Reemplaza con tu usuario de PostgreSQL
        password='melerox1x2',     # Reemplaza con tu contraseña de PostgreSQL
        host='localhost',             # O la dirección de tu servidor PostgreSQL
        port='5432'                   # Puerto por defecto de PostgreSQL
    )
    return conn

# Función para guardar los datos en la base de datos
def save_to_db(data):
    conn = connect_db()
    cur = conn.cursor()
    
    # Crea la tabla si no existe
    cur.execute("""
        CREATE TABLE IF NOT EXISTS satisfaction_data (
            id SERIAL PRIMARY KEY,
            name INTEGER,
            age INTEGER,
            gender INTEGER,
            customer_type INTEGER,
            travel_type INTEGER,
            eco INTEGER,
            eco_plus INTEGER,
            satisfaction INTEGER
        );
    """)
    
    # Inserta los datos en la tabla
    cur.execute("""
        INSERT INTO satisfaction_data (age, gender, customer_type, travel_type, eco, eco_plus, satisfaction)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
    """, (data['age'], data['gender'], data['customer_type'], data['travel_type'], data['eco'], data['eco_plus'], data['satisfaction']))
    
    conn.commit()
    cur.close()
    conn.close()
