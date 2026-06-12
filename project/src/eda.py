import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.config import Config

def plot_distributions(df, save_path="docs/distributions.png"):
    """Визуализация распределений признаков"""
    # Выбираем только числовые колонки
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Исключаем целевую переменную
    num_cols = [col for col in num_cols if col != Config.TARGET_COL]
    
    # Рассчитываем количество графиков
    n_cols = 2
    n_rows = (len(num_cols) + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    
    # Если только один график, axes не список
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}', fontsize=12)
        axes[i].set_xlabel(col)
    
    # Скрываем лишние подграфики
    for i in range(len(num_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✅ Distribution plot saved to {save_path}")

def plot_correlation_matrix(df, save_path="docs/corr_matrix.png"):
    """Матрица корреляций"""
    # Выбираем только числовые колонки
    corr_data = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_data.corr(), 
                annot=True, 
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5)
    plt.title('Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✅ Correlation matrix saved to {save_path}")

def plot_target_distribution(df, save_path="docs/target_dist.png"):
    """Распределение целевой переменной"""
    plt.figure(figsize=(6, 4))
    target_counts = df[Config.TARGET_COL].value_counts()
    target_counts.plot(kind='bar', color=['green', 'red'])
    plt.title('Target Variable Distribution', fontsize=14)
    plt.xlabel('Default Status')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No Default (0)', 'Default (1)'], rotation=0)
    
    # Добавляем проценты
    total = len(df)
    for i, v in enumerate(target_counts.values):
        plt.text(i, v + 100, f'{v/total*100:.1f}%', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"✅ Target distribution saved to {save_path}")

# Функция для проверки
def test_imports():
    print("All functions imported successfully!")