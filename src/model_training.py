import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import load_data, preprocess_data

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.savefig('notebooks/confusion_matrix.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_model(model, scaler, model_path, scaler_path):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Modelo guardado en {model_path}")
    print(f"Scaler guardado en {scaler_path}")

def run_training():
    print("Cargando y preprocesando datos...")
    df = load_data()
    if df is not None:
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        print("Entrenando el modelo...")
        model = train_model(X_train, y_train)
        
        print("Evaluando el modelo...")
        metrics = evaluate_model(model, X_test, y_test)
        
        print("\nMétricas de rendimiento:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        print("\nGuardando el modelo y el scaler...")
        save_model(model, scaler, "models/rf_model.pkl", "models/scaler.pkl")
        
        print("\nEntrenamiento y evaluación del modelo completados.")
        print("Revisa la matriz de confusión en 'notebooks/confusion_matrix.png'")
    else:
        print("No se pudo entrenar el modelo debido a problemas con la carga de datos.")

if __name__ == "__main__":
    run_training()