import subprocess
import time

def start_uvicorn():
    """Inicia el servidor Uvicorn"""
    subprocess.Popen(["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"], shell=True)

def start_streamlit():
    """Inicia la aplicación Streamlit"""
    subprocess.Popen(["python", "-m", "streamlit", "run", "app.py"], shell=True)

if __name__ == "__main__":
    start_uvicorn()
    time.sleep(5)  # Espera un poco para asegurarse de que Uvicorn haya arrancado
    start_streamlit()
