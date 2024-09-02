from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from src.data_preprocessing import load_data, preprocess_data

def load_model_and_scaler():
    model = joblib.load("models/rf_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
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
        'f1': f1,
        'auc': auc
    }

def run_evaluation():
    print("Cargando datos...")
    df = load_data()
    if df is not None:
        print("Preprocesando datos...")
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        print("Cargando modelo...")
        model, _ = load_model_and_scaler()
        
        print("Evaluando modelo...")
        metrics = evaluate_model(model, X_test, y_test)
        
        print("\nMétricas de rendimiento:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        print("\nEvaluación completada. Revisa la matriz de confusión en 'notebooks/confusion_matrix.png'")
    else:
        print("No se pudo evaluar el modelo debido a problemas con la carga de datos.")

if __name__ == "__main__":
    run_evaluation()