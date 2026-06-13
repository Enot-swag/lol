import matplotlib
matplotlib.use('Agg')  # Используем бэкенд без GUI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from src.config import Config

# Настройка стиля
plt.style.use('default')
sns.set_style("whitegrid")

def plot_target_distribution(df, save_path="docs/target_dist.png"):
    """Распределение целевой переменной"""
    os.makedirs("docs", exist_ok=True)
    plt.figure(figsize=(8, 5))
    
    counts = df[Config.TARGET_COL].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    bars = plt.bar(['No Default (0)', 'Default (1)'], counts.values, color=colors)
    plt.title('Target Variable Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    
    # Добавляем проценты
    total = len(df)
    for bar, count in zip(bars, counts.values):
        percentage = count / total * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{count} ({percentage:.1f}%)', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def plot_distributions(df, save_path="docs/distributions.png"):
    """Визуализация распределений признаков (только первые 6)"""
    os.makedirs("docs", exist_ok=True)
    
    # Берём только первые 6 числовых признаков для скорости
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = [col for col in num_cols if col != Config.TARGET_COL][:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        # Убираем выбросы для лучшей визуализации
        data = df[col].clip(upper=df[col].quantile(0.99))
        axes[i].hist(data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].axvline(data.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {data.mean():.2f}')
        axes[i].legend()
    
    # Скрываем лишние подграфики
    for i in range(len(num_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def plot_correlation_matrix(df, save_path="docs/corr_matrix.png"):
    """Матрица корреляций (упрощённая)"""
    os.makedirs("docs", exist_ok=True)
    
    # Берём только числовые колонки
    corr_data = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_data.corr(), dtype=bool))
    sns.heatmap(corr_data.corr(), 
                mask=mask,
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                annot_kws={'size': 8})
    plt.title('Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")