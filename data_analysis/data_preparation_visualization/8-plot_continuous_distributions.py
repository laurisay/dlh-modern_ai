```python
#!/usr/bin/env python3
"""
Module for plotting continuous numerical distributions.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_continuous_distributions(df, columns_to_plot=None):
    """
    Plot distributions of continuous numerical features.

    Each selected column is displayed using a histogram with a KDE
    on the left and a horizontal boxplot on the right.

    Args:
        df: pandas DataFrame containing the data.
        columns_to_plot: Optional list of columns to plot. If None,
            continuous numerical columns are selected automatically.

    Returns:
        None
    """
    if columns_to_plot is None:
        columns_to_plot = [
            column for column in df.select_dtypes(
                include=np.number
            ).columns
            if df[column].nunique() > 2
        ]

    n_cols = len(columns_to_plot)
    fig, axes = plt.subplots(n_cols, 2, figsize=(10, 3 * n_cols))

    if n_cols == 1:
        axes = axes.reshape(1, -1)

    for i, column in enumerate(columns_to_plot):
        data = df[column].dropna()

        axes[i, 0].hist(
            data,
            bins=30,
            density=True,
            alpha=0.7,
            edgecolor='black'
        )

        kde = stats.gaussian_kde(data)
        x_values = np.linspace(data.min(), data.max(), 100)
        axes[i, 0].plot(
            x_values,
            kde(x_values),
            color='red',
            linestyle='--'
        )
        axes[i, 0].set_title(f"{column} Histogram + KDE")

        axes[i, 1].boxplot(data, vert=False)
        axes[i, 1].set_title(f"{column} Boxplot")

    plt.tight_layout()
    plt.savefig("Task_8.png")
    plt.show()
```

