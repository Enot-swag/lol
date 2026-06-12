import pandas as pd
from src.data_preparation import load_data, preprocess_pipeline
from src.eda import plot_distributions, plot_correlation_matrix, plot_target_distribution
from src.config import Config

def main():
    print("="*50)
    print("EDA for Credit Scoring Dataset")
    print("="*50)
    
    # Загрузка данных
    print("\n1. Loading data...")
    df = load_data()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Информация о пропусках
    print("\n2. Missing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if any(missing > 0) else "   No missing values!")
    
    # Статистика
    print("\n3. Basic statistics:")
    print(f"   Target distribution:")
    print(f"   {df[Config.TARGET_COL].value_counts(normalize=True).to_dict()}")
    
    # Визуализации
    print("\n4. Generating visualizations...")
    plot_target_distribution(df)
    plot_distributions(df)
    plot_correlation_matrix(df)
    
    # Проверка предобработки
    print("\n5. Testing preprocessing pipeline...")
    X, y = preprocess_pipeline(df)
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Features names: {X.columns.tolist()}")
    
    print("\n" + "="*50)
    print("✅ EDA completed! Check the 'docs/' folder for plots.")
    print("="*50)

if __name__ == "__main__":
    main()