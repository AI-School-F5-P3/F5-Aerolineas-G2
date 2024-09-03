import bcrypt

# La contraseña que quieres encriptar
password = 'ey'  # Cambia esto a tu contraseña deseada

# Generar el hash de la contraseña
hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Imprimir el hash para que lo puedas copiar al archivo de configuración
print(f'Contraseña encriptada: {hashed.decode("utf-8")}')