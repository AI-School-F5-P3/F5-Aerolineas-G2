import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from jinja2 import Environment, FileSystemLoader

# Función para cargar el modelo entrenado desde un archivo
def load_model(file_path):
    """
    Carga el modelo entrenado desde un archivo pickle.
    
    Args:
    file_path (str): Ruta al archivo del modelo.
    
    Returns:
    object: Modelo cargado.
    """
    return joblib.load(file_path)

# Función para cargar datos desde un archivo CSV
def load_data(file_path):
    """
    Carga los datos desde un archivo CSV.
    
    Args:
    file_path (str): Ruta al archivo CSV.
    
    Returns:
    pandas.DataFrame: DataFrame con los datos cargados.
    """
    return pd.read_csv(file_path)

# Función para preprocesar los datos
def preprocess_data(df):
    """
    Preprocesa los datos, codificando variables categóricas.
    
    Args:
    df (pandas.DataFrame): DataFrame a preprocesar.
    
    Returns:
    pandas.DataFrame: DataFrame preprocesado.
    """
    le = LabelEncoder()
    
    # Codifica la columna 'age_group' si existe
    if 'age_group' in df.columns:
        df['age_group'] = le.fit_transform(df['age_group'])
    
    # Codifica todas las columnas categóricas
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    
    return df

# Función para evaluar el modelo
def evaluate_model(model, X, y):
    """
    Evalúa el modelo y devuelve métricas de rendimiento.
    
    Args:
    model: Modelo entrenado.
    X (pandas.DataFrame): Características de entrada.
    y (pandas.Series): Etiquetas verdaderas.
    
    Returns:
    tuple: Precisión, reporte de clasificación, matriz de confusión, y datos para la curva ROC.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return accuracy, report, cm, fpr, tpr, roc_auc

# Función para evaluar el overfitting
def evaluate_overfitting(model, X_train, y_train, X_test, y_test):
    """
    Evalúa el overfitting comparando el rendimiento en entrenamiento y prueba.
    
    Args:
    model: Modelo entrenado.
    X_train, y_train: Datos de entrenamiento.
    X_test, y_test: Datos de prueba.
    
    Returns:
    tuple: Precisión de entrenamiento, precisión de prueba, y diferencia (overfitting).
    """
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    overfitting = train_accuracy - test_accuracy
    
    return train_accuracy, test_accuracy, overfitting

# Función para crear y guardar la matriz de confusión
def plot_confusion_matrix(cm, output_path):
    """
    Crea y guarda un gráfico de la matriz de confusión.
    
    Args:
    cm (numpy.ndarray): Matriz de confusión.
    output_path (str): Ruta para guardar la imagen.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig(output_path)
    plt.close()

# Función para crear y guardar el gráfico de importancia de características
def plot_feature_importance(model, feature_names, output_path):
    """
    Crea y guarda un gráfico de importancia de características.
    
    Args:
    model: Modelo entrenado.
    feature_names (list): Nombres de las características.
    output_path (str): Ruta para guardar la imagen.
    
    Returns:
    pandas.DataFrame: DataFrame con las importancias de las características.
    """
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Top 15 Características Más Importantes')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return feature_importance

# Función para crear un gráfico interactivo de la curva ROC
def create_roc_curve_plot(fpr, tpr, roc_auc):
    """
    Crea un gráfico interactivo de la curva ROC.
    
    Args:
    fpr (numpy.ndarray): Tasa de falsos positivos.
    tpr (numpy.ndarray): Tasa de verdaderos positivos.
    roc_auc (float): Área bajo la curva ROC.
    
    Returns:
    plotly.graph_objs.Figure: Figura de Plotly con la curva ROC.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    
    return fig

# Función para crear el informe HTML
def create_html_report(accuracy, report, cm, roc_auc, train_accuracy, test_accuracy, overfitting, feature_importance, reports_dir):
    """
    Crea un informe HTML con todas las métricas y resultados.
    
    Args:
    accuracy (float): Precisión del modelo.
    report (str): Reporte de clasificación.
    cm (numpy.ndarray): Matriz de confusión.
    roc_auc (float): Área bajo la curva ROC.
    train_accuracy (float): Precisión en datos de entrenamiento.
    test_accuracy (float): Precisión en datos de prueba.
    overfitting (float): Medida de overfitting.
    feature_importance (pandas.DataFrame): Importancia de las características.
    reports_dir (str): Directorio para guardar el informe.
    """
    env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
    template = env.get_template('report_template.html')
    
    html_content = template.render(
        accuracy=accuracy,
        classification_report=report,
        confusion_matrix=cm.tolist(),
        roc_auc=roc_auc,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        overfitting=overfitting,
        feature_importance=feature_importance.to_dict('records')
    )
    
    with open(os.path.join(reports_dir, "evaluation_report.html"), "w") as f:
        f.write(html_content)

# Bloque principal de ejecución
if __name__ == "__main__":
    # Configuración de rutas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(project_root, "data", "processed", "featured_airlines_data.csv")
    model_path = os.path.join(project_root, "models", "trained_model.pkl")
    reports_dir = os.path.join(project_root, "reports")
    
    # Crear directorio de reportes si no existe
    os.makedirs(reports_dir, exist_ok=True)
    
    # Cargar y preprocesar datos
    print(f"Cargando datos desde {data_path}")
    df = load_data(data_path)
    
    print("Preprocesando datos...")
    df = preprocess_data(df)
    
    # Dividir en características y etiquetas
    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Cargar el modelo entrenado
    print(f"Cargando modelo desde {model_path}")
    model = load_model(model_path)
    
    # Evaluar el modelo
    print("Evaluando modelo...")
    accuracy, report, cm, fpr, tpr, roc_auc = evaluate_model(model, X_test, y_test)
    train_accuracy, test_accuracy, overfitting = evaluate_overfitting(model, X_train, y_train, X_test, y_test)
    
    # Imprimir resultados principales
    print(f"Precisión del modelo en prueba: {accuracy:.4f}")
    print(f"Precisión del modelo en entrenamiento: {train_accuracy:.4f}")
    print(f"Overfitting: {overfitting:.4f}")
    print("\nReporte de Clasificación:")
    print(report)
    
    # Generar visualizaciones
    print("Generando visualizaciones...")
    plot_confusion_matrix(cm, os.path.join(reports_dir, "confusion_matrix.png"))
    feature_importance = plot_feature_importance(model, X.columns, os.path.join(reports_dir, "feature_importance.png"))
    
    roc_fig = create_roc_curve_plot(fpr, tpr, roc_auc)
    roc_fig.write_html(os.path.join(reports_dir, "roc_curve.html"))
    
    # Crear informe HTML
    create_html_report(accuracy, report, cm, roc_auc, train_accuracy, test_accuracy, overfitting, feature_importance, reports_dir)
    
    print("Evaluación completada. Visualizaciones y reporte HTML guardados en el directorio 'reports'.")