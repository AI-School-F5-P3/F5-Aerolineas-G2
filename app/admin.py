import psycopg2
import pandas as pd
import streamlit as st
import bcrypt
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

# Cargar configuración desde .env
load_dotenv()

# Obtener el hash de la contraseña
hashed_password = os.getenv('HASHED_PASSWORD')

def verify_password(password):
    if hashed_password:
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    return False

# Obtener los datos de conexión
dbname = os.getenv('DB_NAME')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')

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
        st.error(f"Error connecting to the database: {e}")
        return None

def data_access():
    if 'access_granted' not in st.session_state:
        st.session_state.access_granted = False

    st.title("Acceso a Datos de Satisfacción")

    if not st.session_state.access_granted:
        # Solicitar la contraseña
        password = st.text_input("Introduce la contraseña para acceder a los datos:", type="password")
        
        if st.button("Acceder"):
            if verify_password(password):
                st.session_state.access_granted = True
                st.success("Acceso concedido.")
            else:
                st.error("Contraseña incorrecta.")
    else:
        # Mostrar los datos si el acceso es concedido
        load_page()

def load_page():
    # Conectar a la base de datos
    conn = get_db_connection()
    if conn is None:
        st.error("No se pudo conectar a la base de datos.")
        return

    # Consultar los datos
    query = "SELECT * FROM customer_predictions"
    df = pd.read_sql(query, conn)
    conn.close()

    # Crear menú de navegación
    st.sidebar.title("Menú de Administración")
    option = st.sidebar.selectbox("Selecciona una opción", [
        "Resumen General", 
        "Análisis por Segmento", 
        "Comparaciones de Predicción", 
        "Distribución de Variables"
    ])

    if option == "Resumen General":
        st.subheader("Datos de Satisfacción del Cliente")
        if not df.empty:
            st.dataframe(df)
            st.write(f"Número total de registros: {len(df)}")
            st.write(f"Promedio de edad: {df['age'].mean():.2f}")
            st.write(f"Promedio de retraso de salida: {df['departure_delay'].mean():.2f} minutos")
            st.write(f"Promedio de retraso de llegada: {df['arrival_delay'].mean():.2f} minutos")
        else:
            st.write("No hay datos disponibles.")

    elif option == "Análisis por Segmento":
        st.subheader("Análisis de Satisfacción por Segmento")
        if not df.empty:
            # Análisis por género
            gender_analysis = df.groupby('gender').agg({
                'probability': 'mean',
                'prediction': 'mean'
            }).reset_index()
            
            st.write("Análisis por Género")
            st.bar_chart(gender_analysis.set_index('gender'))
            
            # Análisis por tipo de cliente
            customer_type_analysis = df.groupby('customer_type').agg({
                'probability': 'mean',
                'prediction': 'mean'
            }).reset_index()
            
            st.write("Análisis por Tipo de Cliente")
            st.bar_chart(customer_type_analysis.set_index('customer_type'))
            
            # Análisis por clase de viaje
            travel_class_analysis = df.groupby('customer_class').agg({
                'probability': 'mean',
                'prediction': 'mean'
            }).reset_index()
            
            st.write("Análisis por Clase de Viaje")
            st.bar_chart(travel_class_analysis.set_index('customer_class'))
        else:
            st.write("No hay datos disponibles para análisis de segmento.")

    elif option == "Comparaciones de Predicción":
        st.subheader("Comparativa entre Predicción y Satisfacción Real")
        if not df.empty:
            # Comparar con la Satisfacción Real
            satisfaction_comparison = df.groupby('real_satisfaction').agg({
                'prediction': 'mean',
                'probability': 'mean'
            }).reset_index()
            
            st.write(satisfaction_comparison)
            st.bar_chart(satisfaction_comparison.set_index('real_satisfaction'))
        else:
            st.write("No hay datos disponibles para comparaciones de predicción.")

    elif option == "Distribución de Variables":
        st.subheader("Distribución de Variables Clave")
        if not df.empty:
            # Histograma de edad
            st.write("Distribución de Edad")
            fig, ax = plt.subplots()
            sns.histplot(df['age'], bins=20, kde=True, ax=ax)
            ax.set_title('Distribución de Edad')
            st.pyplot(fig)
            
            # Histograma de distancia de vuelo
            st.write("Distribución de Distancia de Vuelo")
            fig, ax = plt.subplots()
            sns.histplot(df['flight_distance'], bins=20, kde=True, ax=ax)
            ax.set_title('Distribución de Distancia de Vuelo')
            st.pyplot(fig)
            
            # Gráfico de dispersión: Retraso de salida vs Retraso de llegada
            st.write("Retraso de Salida vs Retraso de Llegada")
            fig, ax = plt.subplots()
            sns.scatterplot(x='departure_delay', y='arrival_delay', data=df, ax=ax)
            ax.set_title('Retraso de Salida vs Retraso de Llegada')
            st.pyplot(fig)
        else:
            st.write("No hay datos disponibles para la distribución de variables.")

if __name__ == "__main__":
    data_access()


