import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import train_agent
from data_loader import generate_unbiased, generate_scenario_1_autoscaler, generate_scenario_2_monitoring
import argparse

def run_mixing_experiment(scenario, num_runs=1):
    print(f"\n--- Starting Mixing Experiment (Scenario {scenario}, {num_runs} Runs) ---")
    
    mix_steps = list(range(0, 51, 5))
    
    # Store results: [Run, Step]
    neural_vio_all = np.zeros((num_runs, len(mix_steps)))
    causal_vio_all = np.zeros((num_runs, len(mix_steps)))
    
    for run in range(num_runs):
        print(f"\n=== Run {run+1}/{num_runs} ===")
        # 1. Generate New Data with random seed based on run
        seed = 1000 + run
        
        scenario_path = f'data/dataset_scenario_{scenario}.csv'
        unbiased_path = f'data/dataset_unbiased.csv'
        
        if scenario == 1:
            generate_scenario_1_autoscaler(seed=seed)
        elif scenario == 2:
            generate_scenario_2_monitoring(threshold=60.0, seed=seed)
        generate_unbiased(seed=seed+999)
        
        df_scenario = pd.read_csv(scenario_path)
        df_unbiased = pd.read_csv(unbiased_path)
        
        # Determine Threshold
        threshold = 100.0 if scenario == 1 else 60.0
        
        for i, x in enumerate(mix_steps):
            print(f"  Biased + {x} Unbiased (Threshold: {threshold})...")
            
            # Mix Data
            df_mix = pd.concat([df_scenario, df_unbiased.iloc[:x]])
            temp_file = f'data/dataset_mixed_temp_run{run}.csv'
            df_mix.to_csv(temp_file, index=False)
            
            # Train Neural
            try:
                # Suppress output? Maybe reduce train_steps further for speed if N runs is high
                # Keeping 2000 steps is good for convergence.
                model_n = train_agent.train_agent(temp_file, L_threshold=threshold, agent_type='neural', train_steps=2000)
                metrics_n = train_agent.evaluate_agent(model_n, L_threshold=threshold, n_episodes=50)
                neural_vio_all[run, i] = metrics_n['violations']
            except Exception as e:
                # print(f"Neural Failed: {e}")
                neural_vio_all[run, i] = 100.0

            # Train Causal
            try:
                model_c = train_agent.train_agent(temp_file, L_threshold=threshold, agent_type='causal', train_steps=2000)
                metrics_c = train_agent.evaluate_agent(model_c, L_threshold=threshold, n_episodes=50)
                causal_vio_all[run, i] = metrics_c['violations']
            except Exception as e:
                # print(f"Causal Failed: {e}")
                causal_vio_all[run, i] = 100.0
                
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # 3. Aggregation
    n_mean = np.mean(neural_vio_all, axis=0)
    n_std = np.std(neural_vio_all, axis=0)
    c_mean = np.mean(causal_vio_all, axis=0)
    c_std = np.std(causal_vio_all, axis=0)
    
    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    # Neural
    plt.plot(mix_steps, n_mean, 'r-o', label='Neural Agent', linewidth=2)
    plt.fill_between(mix_steps, n_mean - n_std, n_mean + n_std, color='red', alpha=0.15)
    
    # Causal
    plt.plot(mix_steps, c_mean, 'g-s', label='Causal Agent', linewidth=2)
    plt.fill_between(mix_steps, c_mean - c_std, c_mean + c_std, color='green', alpha=0.15)
    
    plt.xlabel('Number of Unbiased Samples Added')
    plt.ylabel('Violation Rate (%)')
    plt.title(f'Agent Robustness vs Data Quality (Scenario {scenario}, {num_runs} Runs)')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    outfile = f'plots/mixing_experiment_multirun_scenario_{scenario}.png'
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(outfile)
    
    print(f"\nMulti-Run Experiment Complete. Saved {outfile}")

import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, default=1, help='Scenario to evaluate')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs for averaging')
    args = parser.parse_args()
    run_mixing_experiment(args.scenario, args.runs)
