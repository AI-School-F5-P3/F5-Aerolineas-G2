from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import joblib
from src.data_preprocessing import load_data, preprocess_data

def tune_hyperparameters(X_train, y_train):
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(5, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=100, cv=5, random_state=42, n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    print("Mejores hiperparámetros encontrados:")
    print(random_search.best_params_)
    
    return random_search.best_estimator_

def run_hyperparameter_tuning():
    print("Cargando datos...")
    df = load_data()
    if df is not None:
        print("Preprocesando datos...")
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        print("Iniciando ajuste de hiperparámetros...")
        best_model = tune_hyperparameters(X_train, y_train)
        
        print("Guardando el mejor modelo...")
        joblib.dump(best_model, "models/rf_model_tuned.pkl")
        print("Modelo optimizado guardado en 'models/rf_model_tuned.pkl'")
    else:
        print("No se pudo realizar el ajuste de hiperparámetros debido a problemas con la carga de datos.")

if __name__ == "__main__":
    run_hyperparameter_tuning()