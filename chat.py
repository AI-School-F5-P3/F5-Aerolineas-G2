import matplotlib.pyplot as plt
import networkx as nx

def create_correct_diagram():
    # Crear un gráfico dirigido
    G = nx.DiGraph()

    # Agregar los nodos
    G.add_node("Datos Originales", pos=(0, 4))
    G.add_node("80% Entrenamiento", pos=(0, 3))
    G.add_node("20% Evaluación Final", pos=(2, 3))
    G.add_node("Validación Cruzada\nAjuste de Hiperparámetros", pos=(0, 2))
    G.add_node("Validación Cruzada\nEvaluación con Hiperparámetros", pos=(0, 1))
    G.add_node("Evaluar con\n20% Reservado", pos=(0, 0))

    # Agregar las aristas (conexiones)
    G.add_edges_from([
        ("Datos Originales", "80% Entrenamiento"),
        ("Datos Originales", "20% Evaluación Final"),
        ("80% Entrenamiento", "Validación Cruzada\nAjuste de Hiperparámetros"),
        ("Validación Cruzada\nAjuste de Hiperparámetros", "Validación Cruzada\nEvaluación con Hiperparámetros"),
        ("Validación Cruzada\nEvaluación con Hiperparámetros", "Evaluar con\n20% Reservado"),
        ("20% Evaluación Final", "Evaluar con\n20% Reservado")  # Flecha cruzada
    ])

    # Obtener posiciones de los nodos
    pos = nx.get_node_attributes(G, 'pos')

    # Crear la figura
    plt.figure(figsize=(10, 6))

    # Dibujar los nodos con colores y bordes
    nx.draw_networkx_nodes(G, pos, node_color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightblue'],
                           node_size=3000, edgecolors='black', linewidths=1.5)

    # Dibujar las etiquetas
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    # Dibujar las aristas con flechas
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', width=2)

    # Ajustar los márgenes y mostrar
    plt.title("Proceso de Validación Cruzada y Evaluación de Modelo (Camino Único)", fontsize=14)
    plt.tight_layout()
    plt.axis('off')
    plt.show()

# Llamar la función para generar el diagrama
create_correct_diagram()
