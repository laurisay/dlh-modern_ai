#!/usr/bin/env python3
"""
Visualize missing values in a DataFrame.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """
    Visualizes missing values in a DataFrame.
    """
    plt.figure(figsize=(12, 8))
    
    # Trouver les positions des valeurs manquantes
    rows, cols = np.where(df.isnull())
    
    # Créer le scatter plot
    plt.scatter(rows, cols, marker='|', s=100)
    
    # Configurer l'axe Y avec les noms de colonnes
    plt.yticks(np.arange(len(df.columns)), df.columns)
    
    # Inverser l'axe Y pour avoir Churn en haut
    plt.gca().invert_yaxis()
    
    # Ajouter le titre demandé par la référence
    plt.title('Missingness Plot')
    
    # Ajuster et afficher
    plt.tight_layout()
    plt.show()
