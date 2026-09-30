import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import argparse

# Import new modules
from envs.autoscaler_env import real_physics
from agents.neural_agent import NeuralAgent
from agents.causal_agent import CausalAgent

# --- Training Loop ---
def train_agent(data_path, L_threshold, agent_type='neural', train_steps=5000):
    """
    Main entry point for training agents.
    Dispatches to NeuralAgent or CausalAgent.
    Returns the trained SB3 model.
    """
    # print(f"\n--- Training {agent_type.upper()} Agent on {data_path} (Threshold: {L_threshold}) ---")
    
    if agent_type == 'neural':
        agent = NeuralAgent(L_threshold=L_threshold)
        model = agent.train(data_path, train_steps=train_steps)
    elif agent_type == 'causal':
        agent = CausalAgent(L_threshold=L_threshold)
        model = agent.train(data_path, train_steps=train_steps)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    return model

# --- Evaluation ---
def evaluate_agent(model, L_threshold, n_episodes=20):
    # print("\n Evaluating on REAL PHYSICS...")
    
    t_samples = np.linspace(500, 1500, n_episodes)
    
    total_violations = 0
    total_cost = 0
    
    for t_val in t_samples:
        action, _ = model.predict(np.array([t_val], dtype=np.float32))
        
        # Ensure action is scalar
        if isinstance(action, np.ndarray):
            action = action.item()
            
        r_val = action + 1
        
        # Real Physics Check
        real_l = real_physics(t_val, r_val)
        violation = 1 if real_l > L_threshold else 0
        
        if violation:
            # print(f"T={t_val:.0f} -> R={r_val} -> L={real_l:.1f} [VIOLATION]")
            pass
        else:
            pass
            
        total_violations += violation
        total_cost += r_val
        
    avg_cost = total_cost/n_episodes
    violation_rate = (total_violations/n_episodes) * 100.0
    print(f"Results: Violations={total_violations}/{n_episodes} ({violation_rate:.1f}%), Avg Cost={avg_cost:.1f}")
    
    return {'violations': violation_rate, 'cost': avg_cost}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DRL Agent (Neural vs Causal)")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2], help="Scenario 1 (Autoscaler) or 2 (Monitoring)")
    parser.add_argument("--agent", type=str, default='neural', choices=['neural', 'causal'], help="Agent type")
    parser.add_argument("--train_steps", type=int, default=5000, help="Training timesteps")
    parser.add_argument("--threshold", type=float, default=100.0, help="Latency Threshold (SLA)")
    args = parser.parse_args()
    
    filename = f"data/dataset_scenario_{args.scenario}.csv"
    
    # Adjust threshold default for Scenario 2 if not strict?
    # User uses 60.0 for generation in Scenario 2.
    # We'll use args.threshold
    
    # Train
    model = train_agent(filename, L_threshold=args.threshold, agent_type=args.agent, train_steps=args.train_steps)
    
    # Evaluate
    evaluate_agent(model, L_threshold=args.threshold)
