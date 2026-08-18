#!/usr/bin/env python3
"""Function to visualize missing values in a DataFrame."""

import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """Visualizes missing values in a DataFrame as a scatter plot.

    Args:
        df: pandas DataFrame to analyze.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))

    # Créer une matrice binaire: 1 pour les valeurs manquantes, 0 pour les valeurs non-manquantes
    missing_matrix = df.isnull().values

    # Obtenir les indices des valeurs manquantes
    rows, cols = np.where(missing_matrix)

    # Récupérer les noms des colonnes pour les y-ticks
    column_names = df.columns.tolist()

    # Créer un mapping des noms de colonnes vers des indices numériques
    col_to_idx = {col: i for i, col in enumerate(column_names)}

    # Convertir les noms de colonnes en indices numériques pour le tracé
    y_positions = [col_to_idx[column_names[col]] for col in cols]

    # Tracer chaque valeur manquante comme un point
    plt.scatter(rows, y_positions, marker='|', s=100, color='blue')

    # Configurer les axes
    plt.xlabel('Row Index')
    plt.ylabel('Columns')
    plt.title('Missing Values Visualization')

    # Définir les y-ticks avec les noms de colonnes
    plt.yticks(range(len(column_names)), column_names)

    # Inverser l'axe y pour que la première colonne soit en haut
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()
