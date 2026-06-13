from src.data_preparation import load_data, preprocess_pipeline
from src.eda import plot_distributions, plot_correlation_matrix, plot_target_distribution
from src.config import Config

def main():
    print("="*50)
    print("EDA FOR CREDIT SCORING")
    print("="*50)
    
    print("\n[1/4] Loading data...")
    df = load_data()
    
    print("\n[2/4] Data info:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Missing: {df.isnull().sum().sum()}")
    print(f"   Target distribution: {df[Config.TARGET_COL].value_counts(normalize=True).to_dict()}")
    
    print("\n[3/4] Generating plots...")
    plot_target_distribution(df)
    plot_distributions(df)
    plot_correlation_matrix(df)
    
    print("\n[4/4] Testing preprocessing...")
    X, y = preprocess_pipeline(df)
    print(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    print("\n" + "="*50)
    print("✅ EDA COMPLETED! Check docs/ folder for plots.")
    print("="*50)

if __name__ == "__main__":
    main()