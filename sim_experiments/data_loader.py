import pandas as pd
import numpy as np
import os

def generate_scenario_1_autoscaler(n_samples=5000, seed=42):
    """
    Scenario 1: Autoscaler Confounding Bias.
    T -> L <- R (Physical)
    T -> R (Autoscaler Policy)
    Outcome: Spurious positive correlation between T and R.
    """
    print("Generating Scenario 1: Autoscaler Bias...")
    if seed is not None:
        np.random.seed(seed)
    
    # 1. Exogenous: Traffic T
    # Range: 500 to 1500 rps
    T = np.random.uniform(500, 1500, n_samples)
    
    # 1. Confounder: Resources R (determined by Autoscaler)
    # Autoscaler Logic: R = T / 40.0 + Noise. (Target L ~ 50)
    # This creates T -> R
    # Use Uniform Noise for LiNGAM identifiability
    R = (T / 40.0) + np.random.uniform(-1.0, 1.0, n_samples)
    R = np.clip(R, 1, 40) # Min 1, Max 40 replicas
    
    # 3. Outcome: Latency L
    # Physics: Linear Approximation for LiNGAM compatibility
    # L = 0.5 * T - 20.0 * R + 50.0 + Noise
    # Higher T -> Higher L. Higher R -> Lower L.
    L = 0.5 * T - 20.0 * R + 50.0 + np.random.uniform(-5.0, 5.0, n_samples)
    
    df = pd.DataFrame({'T': T, 'R': R, 'L': L})
    
    output_path = 'data/dataset_scenario_1.csv'
    df.to_csv(output_path, index=False)
    # print(f"Saved {output_path}. Correlation T-R: {df['T'].corr(df['R']):.3f}")
    return output_path

def generate_scenario_2_monitoring(threshold, n_samples=5000, seed=42):
    """
    Scenario 2: Successful Monitoring Selection Bias.
    T -> L <- R (Physical)
    T, R Independent (Random Policy)
    Condition: L < Threshold (Success)
    (Spurious correlation T-R).
    """
    print("Generating Scenario 2: Monitoring Bias...")
    if seed is not None:
        np.random.seed(seed)
    
    # Generate many outcomes, filter for success
    n_total = n_samples * 5
    
    # 1. Independent T and R
    T = np.random.uniform(500, 1500, n_total)
    R = np.random.uniform(10, 40, n_total) # Range to match linear scale
    
    # 2. Outcome: Latency L (Linear)
    L = 0.5 * T - 20.0 * R + 50.0 + np.random.uniform(-5.0, 5.0, n_total)
    
    df_all = pd.DataFrame({'T': T, 'R': R, 'L': L})
    
    # 3. Selection Bias: Monitoring only records if L < Threshold
    # We want to select successes (Low Latency).
    # Mean L? T=1000 -> 500. R=25 -> 500. L=50.
    #threshold = 60.0 # Selects ~50%?
    df_observed = df_all[df_all['L'] < threshold].copy()
    
    # Subsample to requested size
    if len(df_observed) > n_samples:
        df_observed = df_observed.sample(n_samples, random_state=(seed if seed is not None else 42))
    
    output_path = 'data/dataset_scenario_2.csv'
    df_observed.to_csv(output_path, index=False)
    # print(f"Saved {output_path}. Condition: L < {threshold}. Correlation T-R (Spurious): {df_observed['T'].corr(df_observed['R']):.3f}")
    return output_path

def generate_unbiased(n_samples=5000, seed=999):
    """
    Random Policy: Unbiased Data.
    T, R Independent.
    Used for mixing experiments.
    """
    print("Generating Unbiased Data...")
    if seed is not None:
        np.random.seed(seed)
    
    T = np.random.uniform(500, 1500, n_samples)
    R = np.random.uniform(1, 40, n_samples) # Random allocation covering full space
    
    # Physics (Same as others)
    L = 0.5 * T - 20.0 * R + 50.0 + np.random.uniform(-5.0, 5.0, n_samples)
    
    df = pd.DataFrame({'T': T, 'R': R, 'L': L})
    output_path = 'data/dataset_unbiased.csv'
    df.to_csv(output_path, index=False)
    # print(f"Saved {output_path}. Correlation T-R: {df['T'].corr(df['R']):.3f}")
    return output_path

if __name__ == "__main__":
    generate_scenario_1_autoscaler()
    generate_scenario_2_monitoring()
