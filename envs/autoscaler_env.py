import gymnasium as gym
from gymnasium import spaces
import numpy as np

# --- 1. Real Physics (Ground Truth for Evaluation) ---
def real_physics(T, R):
    """
    Tru Dynamics: L = 0.5 * T - 20.0 * R + 50.0 + Noise
    """
    return 0.5 * T - 20.0 * R + 50.0

# --- 2. Gym EnvironmentWrapper ---
class AutoscalerEnv(gym.Env):
    def __init__(self, world_model, L_threshold=100, mode='neural'):
        super().__init__()
        self.world_model = world_model
        self.mode = mode
        self.L_threshold = L_threshold
        
        # Observation: [T] (Exogenous Context)
        self.observation_space = spaces.Box(low=0, high=2000, shape=(1,), dtype=np.float32)
        
        # Action: R (discrete 1..40)
        self.action_space = spaces.Discrete(40) # 0 maps to 1 replica, 39 to 40
        
        self.state_t = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Sample T from distribution (500 to 1500)
        self.state_t = np.random.uniform(500, 1500)
        return np.array([self.state_t], dtype=np.float32), {}
        
    def step(self, action):
        # Action 0..39 -> Replicas 1..40
        r_val = action + 1
        t_val = self.state_t
        
        # Use World Model to predict outcome L
        # World Model expects (T, R) input
        
        # Handle different predict signatures if necessary (though we should standardize)
        # Assuming both models accept (t_scalar, r_scalar) and return predictions
        if self.mode == 'neural':
            import torch
            # Neural model expects tensors
            t_tens = torch.tensor([t_val], dtype=torch.float32)
            r_tens = torch.tensor([r_val], dtype=torch.float32)
            with torch.no_grad():
                l_pred = self.world_model(t_tens, r_tens).item()
        else:
            # Causal model expects numpy/scalars
            l_pred = self.world_model.predict(t_val, r_val)[0]
            
        # Compute Reward based on World Model's belief
        # Objective: Minimize Cost (R) s.t. L < Threshold
        # Cost: - R_val
        # SLA Penalty: If L > Threshold, -1000
        cost = r_val
        penalty = 1000.0 if l_pred > self.L_threshold else 0.0
        reward = -(cost + penalty)
        
        terminated = True # Contextual Bandit (1 step)
        truncated = False
        
        return np.array([t_val], dtype=np.float32), reward, terminated, False, {'latency': l_pred, 'replicas': r_val}
