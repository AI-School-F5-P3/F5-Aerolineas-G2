import pandas as pd

# Ruta correcta al archivo CSV
df = pd.read_csv('mi_carpeta/airline_passenger_satisfaction.csv')

# Mostrar las primeras filas para verificar la carga
print(df.head())
