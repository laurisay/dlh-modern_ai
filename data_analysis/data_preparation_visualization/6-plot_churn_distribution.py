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
    
    # Get value counts and create bar plot
    churn_counts = df['Churn'].value_counts()
    
    # Create bar plot with specific colors
    colors = ['skyblue' if label == 'No' else 'salmon' for label in churn_counts.index]
    bars = plt.bar(churn_counts.index, churn_counts.values, color=colors)
    
    # Customize the plot
    plt.title('Churn Distribution', fontsize=16, pad=20)
    plt.xlabel('Churn', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=11)
    
    # Set y-axis to show full numbers
    plt.ticklabel_format(style='plain', axis='y')
    
    # Display the plot
    plt.tight_layout()
    plt.show()
