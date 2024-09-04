import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
# Asumimos que los datos están en un archivo CSV llamado 'airline_satisfaction.csv' en el directorio raíz
df = pd.read_csv('airline_passenger_satisfaction.csv')

# Visualizar las primeras filas y la información del DataFrame
print(df.head())
print(df.info())

# Verificar valores nulos
print(df.isnull().sum())

# Estadísticas descriptivas
print(df.describe())

# Convertir variables categóricas a numéricas
df = pd.get_dummies(df, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'], drop_first=True)

# Separar características y variable objetivo
X = df.drop('satisfaction', axis=1)
y = df['satisfaction']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Imputamos los valores nulos con la mediana
df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median())

# ----------------------------------

from sklearn.preprocessing import LabelEncoder

# Separar variables numéricas y categóricas
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# # Incluir la variable codificada de satisfacción
# numeric_columns = numeric_columns.append(pd.Index(['satisfaction_encoded']))

# Codificar la variable objetivo
le = LabelEncoder()
df['satisfaction_encoded'] = le.fit_transform(df['satisfaction'])

# Análisis de variables numéricas
plt.figure(figsize=(20, 15))
df.hist(bins=30, figsize=(20, 15))
plt.tight_layout()
plt.show()

# Correlación entre variables numéricas con heatmap
correlation_matrix = df[numeric_columns].corr()
# Generar la máscara para ocultar la mitad superior
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Correlation Heatmap of Numeric Variables')
plt.show()

# Análisis de variables categóricas
# for col in categorical_columns:
#     plt.figure(figsize=(10, 6))
#     df[col].value_counts().plot(kind='bar')
#     plt.title(f'Distribution of {col}')
#     plt.ylabel('Count')
#     plt.xticks(rotation=45)
#     plt.show()
# ----------------------------------
# Definir el número de columnas y filas
n_cols = 2
n_rows = (len(categorical_columns) + 1) // 2  # Calcular el número de filas necesarias

# Crear subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()  # Aplanar los ejes para facilitar el manejo

# Graficar cada columna categórica
for i, col in enumerate(categorical_columns):
    df[col].value_counts().plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_ylabel('Count')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

# Eliminar gráficos sobrantes si hay un número impar de columnas
for j in range(len(categorical_columns), len(axes)):
    fig.delaxes(axes[j])

# Ajustar el espaciado entre gráficos
plt.tight_layout()
plt.show()

# Relación entre variables categóricas y la satisfacción
# for col in categorical_columns:
#     if col != 'satisfaction':
#         plt.figure(figsize=(12, 6))
#         sns.countplot(data=df, x=col, hue='satisfaction')
#         plt.title(f'Satisfaction by {col}')
#         plt.xticks(rotation=45)
#         plt.show()
# ----------------------------------
# Filtrar las variables categóricas (excluyendo 'satisfaction')
filtered_columns = [col for col in categorical_columns if col != 'satisfaction']

# Definir el número de columnas y filas
n_cols = 2
n_rows = (len(filtered_columns) + 1) // 2  # Calcular el número de filas necesarias

# Crear subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()  # Aplanar los ejes para facilitar el manejo

# Graficar cada columna categórica (excepto 'satisfaction')
for i, col in enumerate(filtered_columns):
    sns.countplot(data=df, x=col, hue='satisfaction', ax=axes[i])
    axes[i].set_title(f'Satisfaction by {col}')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

# Eliminar gráficos sobrantes si hay un número impar de columnas
for j in range(len(filtered_columns), len(axes)):
    fig.delaxes(axes[j])

# Ajustar el espaciado entre gráficos
plt.tight_layout()
plt.show()

