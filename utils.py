import psycopg2
import configparser
import streamlit as st
import pandas as pd
import bcrypt

# Leer la configuración desde el archivo config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Obtener el hash de la contraseña
hashed_password = config['auth'].get('hashed_password')

def verify_password(password):
    if hashed_password:
        print(f"Hash en config: {hashed_password}") 
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    return False

# Obtener los datos de conexión
db_config = config['database']
dbname = db_config.get('dbname')
user = db_config.get('user')
password = db_config.get('password')
host = db_config.get('host')
port = db_config.get('port')

def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

# Página para ver los datos
def data_access():
    st.title("Acceso a Datos de Satisfacción")
    
    # Solicitar la contraseña
    password = st.text_input("Introduce la contraseña para acceder a los datos:", type="password")
    
    if st.button("Acceder"):
        if verify_password(password):
            st.success("Acceso concedido.")
            # Aquí se debe llamar a la función para cargar los datos
            load_page()  # Llama a la función desde utils.py para cargar los datos
        else:
            st.error("Contraseña incorrecta.")

def load_page():
    # Conectar a la base de datos
    conn = get_db_connection()
    if conn is None:
        st.error("No se pudo conectar a la base de datos.")
        return

    # Consultar los datos
    query = "SELECT * FROM satisfaction_data"
    df = pd.read_sql(query, conn)
    
    # Mostrar datos en la app
    st.write("Datos de Satisfacción del Cliente")
    st.dataframe(df)
    
    # Mostrar algunas métricas simples
    st.write(f"Número total de registros: {len(df)}")
    st.write(f"Promedio de edad: {df['age'].mean():.2f}")
    st.write(f"Promedio de retraso de salida: {df['departure_delay'].mean():.2f} minutos")
    st.write(f"Promedio de retraso de llegada: {df['arrival_delay'].mean():.2f} minutos")
    
    conn.close()