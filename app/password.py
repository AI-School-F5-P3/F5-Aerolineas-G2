# password_manager.py

import bcrypt

def generate_password_hash(password: str) -> str:
    """Genera un hash para la contraseña proporcionada."""
    # Generar un salt
    salt = bcrypt.gensalt()
    # Crear el hash de la contraseña
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')

def verify_password(stored_hash: str, password: str) -> bool:
    """Verifica si la contraseña proporcionada coincide con el hash almacenado."""
    # Comparar la contraseña proporcionada con el hash almacenado
    return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))

if __name__ == "__main__":
    # Ejemplo de uso
    password = "sergio123"
    hashed = generate_password_hash(password)
    print(f"Contraseña Hashed: {hashed}")
    
    # Verificación
    is_correct = verify_password(hashed, password)
    print(f"La contraseña es correcta: {is_correct}")

