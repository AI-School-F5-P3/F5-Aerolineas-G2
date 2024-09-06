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

# Borramos la primera columna que se crea al cargar el archivo CSV
columnas_a_borrar = ['Unnamed: 0', 'id']
df = df.drop(columnas_a_borrar, axis=1)

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
# ---
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
# ------
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

# ----------------------------------

# Relación entre variables categóricas y satisfacción
# Separar variables categóricas
categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('satisfaction', errors='ignore')  # Excluir la variable de satisfacción

# Crear un DataFrame para almacenar las proporciones
proportions = pd.DataFrame()

# Calcular proporciones de satisfacción para cada categoría
for col in categorical_columns:
    # Crear una tabla de contingencia
    contingency_table = pd.crosstab(df[col], df['satisfaction_encoded'])
    # Calcular proporciones de satisfacción
    proportions[col] = contingency_table.apply(lambda x: x / x.sum(), axis=1).iloc[:, 1]  # Proporción de 'satisfied'
    
# Mostrar proporciones
print(proportions)

import scipy.stats as stats

# Crear una lista para almacenar los resultados de la prueba de Chi-cuadrado
chi2_results = []

# Realizar la prueba de Chi-cuadrado para cada variable categórica
for col in categorical_columns:
    contingency_table = pd.crosstab(df[col], df['satisfaction_encoded'])
    chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
    chi2_results.append({'Variable': col, 'Chi2 Stat': chi2_stat, 'P-Value': p_value})

# Convertir la lista de resultados a un DataFrame
chi2_results_df = pd.DataFrame(chi2_results)

# Mostrar los resultados de la prueba de Chi-cuadrado
print(chi2_results_df)

# ----------------------------------

# Correlación en número entre variables numéricas y satisfacción
# Separar variables numéricas y la variable codificada de satisfacción
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
numeric_columns = numeric_columns.drop('satisfaction_encoded', errors='ignore')  # Excluir la variable de satisfacción

# Calcular la correlación entre la variable de satisfacción y cada variable numérica
correlations = df[numeric_columns].apply(lambda x: x.corr(df['satisfaction_encoded']))

# Mostrar las correlaciones
print(correlations)

# ----------------------------------

# Relación de las variables numéricas con la satisfacción
# Definir el número de columnas y filas para los subplots
n_cols = 2
n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Calcular el número de filas necesarias

# Crear subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()  # Aplanar los ejes para facilitar el manejo

# Graficar cada variable numérica en relación con la satisfacción
for i, col in enumerate(numeric_columns):
    if col != 'satisfaction_encoded':  # No graficar la variable de codificación en sí misma
        # Graficar la variable numérica en relación con la satisfacción
        sns.barplot(x='satisfaction', y=col, data=df, ax=axes[i])
        axes[i].set_title(f'{col} by Satisfaction')
        axes[i].set_ylabel(f'{col}')
        axes[i].set_xlabel('Satisfaction')
    else:
        axes[i].axis('off')  # Desactivar los gráficos vacíos si hay menos variables numéricas

# Eliminar gráficos sobrantes si hay un número impar de variables numéricas
for j in range(len(numeric_columns), len(axes)):
    axes[j].axis('off')

# Ajustar el espaciado entre gráficos
plt.tight_layout()
plt.show()

# ----------------------------------
# Relación enetre variables numércias y satisfacción con gráfica de caja y bigotes para detectar outliers

# Definir el número de filas y columnas (2 gráficos por fila)
n_cols = 2
n_rows = len(numeric_columns) // n_cols + (len(numeric_columns) % n_cols > 0)

# Crear subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

# Aplanar los ejes para que sean más fáciles de manejar
axes = axes.flatten()

# Graficar cada columna numérica
for i, col in enumerate(numeric_columns):
    sns.boxplot(data=df, x='satisfaction', y=col, ax=axes[i])
    axes[i].set_title(f'{col} by Satisfaction')
    axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)

# Eliminar gráficos sobrantes si hay un número impar de columnas
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Ajustar el espaciado entre gráficos
plt.tight_layout()
plt.show()

# ----------------------------------

# Este código entrena el modelo, realiza predicciones, evalúa su rendimiento y visualiza la importancia de las características.

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Borrado de las dos primeras columnas
# columnas_a_borrar = ['Unnamed: 0', 'id']
# df = df.drop(columnas_a_borrar, axis=1)

# Imputar los valores nulos con la media
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Escalar las características imputadas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# 1. Random Forest: Crear y entrenar el modelo
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
# Realizar predicciones
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# 2. Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# 3. Support Vector Machine (SVM)
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

# Evaluar y comparar los modelos
print("Evaluación del modelo Random Forest:")
print(classification_report(y_test, y_pred_rf))
print("Matriz de Confusión para Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))
print("AUC-ROC para Random Forest:")
print(roc_auc_score(y_test, y_pred_proba_rf))

print("\nEvaluación del modelo Logistic Regression:")
print(classification_report(y_test, y_pred_lr))
print("Matriz de Confusión para Logistic Regression:")
print(confusion_matrix(y_test, y_pred_lr))
print("AUC-ROC para Logistic Regression:")
print(roc_auc_score(y_test, y_pred_proba_lr))

print("\nEvaluación del modelo SVM:")
print(classification_report(y_test, y_pred_svm))
print("Matriz de Confusión para SVM:")
print(confusion_matrix(y_test, y_pred_svm))
print("AUC-ROC para SVM:")
print(roc_auc_score(y_test, y_pred_proba_svm))

# Comparación de Overfitting
rf_train_score = rf_model.score(X_train_scaled, y_train)
rf_test_score = rf_model.score(X_test_scaled, y_test)
rf_overfitting = rf_train_score - rf_test_score

lr_train_score = lr_model.score(X_train_scaled, y_train)
lr_test_score = lr_model.score(X_test_scaled, y_test)
lr_overfitting = lr_train_score - lr_test_score

svm_train_score = svm_model.score(X_train_scaled, y_train)
svm_test_score = svm_model.score(X_test_scaled, y_test)
svm_overfitting = svm_train_score - svm_test_score

print("\nOverfitting Random Forest:", rf_overfitting)
print("Overfitting Logistic Regression:", lr_overfitting)
print("Overfitting SVM:", svm_overfitting)

# Calcular y visualizar la importancia de las características
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Características Más Importantes')
plt.show()

# ----------------------------------

