#!/usr/bin/env python3
"""
Plot churn distribution
"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """
    Visualizes churn class distribution as a bar plot
    
    Args:
        df: pandas DataFrame with a Churn column
    
    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    
    # Get value counts and ensure consistent order
    churn_counts = df['Churn'].value_counts()
    
    # Reindex to ensure 'No' comes first if it exists
    if 'No' in churn_counts.index and 'Yes' in churn_counts.index:
        churn_counts = churn_counts[['No', 'Yes']]
    
    # Set colors
    colors = []
    for label in churn_counts.index:
        if label == 'No':
            colors.append('skyblue')
        elif label == 'Yes':
            colors.append('salmon')
        else:
            colors.append('gray')
    
    # Create bar plot
    bars = plt.bar(churn_counts.index, churn_counts.values, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 str(int(height)),
                 ha='center', va='bottom')
    
    # Set labels and title
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.title('Churn Distribution')
    
    # Display the plot
    plt.tight_layout()
    plt.show()
