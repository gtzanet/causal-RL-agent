# ---- DQN agents: train 3 agents (train-set, eval-train-set, causal-train-set) ----
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict


class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_1=64, hidden_dim_2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action_idx),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.asarray(s2, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        )


class QLearningAgent:
    # Keeps historical class name, but implementation follows DQN training from the paper.
    def __init__(
        self,
        env,
        alpha=0.001,
        gamma=0.90,
        epsilon_start=1.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
        replay_buffer_size=50000,
        batch_size=64,
        replay_warmup_epochs=100,
        target_update_epochs=5,
        updates_per_step=1,
        eval_interval_epochs=50,
        eval_episodes=3,
        feature_indices=None,
    ):
        self.env = env
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon = float(epsilon_start)

        self.n_actions = len(env.actions)
        self.feature_indices = None if feature_indices is None else list(feature_indices)
        self.state_dim = len(self._state_view(env.reset()))

        self.q_network = DQNNetwork(self.state_dim, self.n_actions)
        self.target_network = DQNNetwork(self.state_dim, self.n_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = int(batch_size)
        self.replay_warmup_epochs = int(replay_warmup_epochs)
        self.target_update_epochs = int(target_update_epochs)
        self.updates_per_step = int(updates_per_step)
        self.eval_interval_epochs = int(eval_interval_epochs)
        self.eval_episodes = int(eval_episodes)

        # Backward-compatible lightweight table view for downstream diagnostics.
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))
        self.eval_trace = []

    def _state_view(self, state):
        arr = np.asarray(state, dtype=np.float32)
        if self.feature_indices is None:
            return arr
        return arr[self.feature_indices]

    def _select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s_tensor = torch.tensor(self._state_view(state), dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network(s_tensor)
            return int(q_values.argmax(dim=1).item())

    def greedy_action(self, state):
        return self._select_action(state, training=False)

    def _update_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        s_batch = torch.tensor(states)
        a_batch = torch.tensor(actions).unsqueeze(1)
        r_batch = torch.tensor(rewards).unsqueeze(1)
        s2_batch = torch.tensor(next_states)
        d_batch = torch.tensor(dones).unsqueeze(1)

        q_current = self.q_network(s_batch).gather(1, a_batch)
        with torch.no_grad():
            q_target_next = self.target_network(s2_batch).max(1)[0].unsqueeze(1)
            target = r_batch + self.gamma * q_target_next * (1.0 - d_batch)

        loss = self.loss_fn(q_current, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def _run_episode(self, training=True):
        state = self.env.reset()
        done = False
        G = 0.0
        ep_violations = 0
        ep_steps = 0

        while not done:
            a_idx = self._select_action(state, training=training)
            next_state, reward, done, info = self.env.step(a_idx)
            G += reward
            ep_violations += int(info.get("violation", 0))

            if training:
                # Store the transition
                self.replay_buffer.push(
                    self._state_view(state),
                    a_idx,
                    reward,
                    self._state_view(next_state),
                    float(done),
                )
                # Keep lightweight Q-table view up to date for diagnostic tasks.
                h_s = tuple(np.round(np.asarray(state, dtype=float), decimals=2).tolist())
                self.q_table[h_s][a_idx] = reward

                # Execute updates if buffer is sufficiently warmed up
                if len(self.replay_buffer) >= self.batch_size:
                    for _ in range(self.updates_per_step):
                        self._update_network()

            state = next_state
            ep_steps += 1

        return G, ep_violations / max(1, ep_steps)

    def _evaluate_greedy(self, episodes=None):
        n_eval = self.eval_episodes if episodes is None else int(episodes)
        returns = []
        violations = []
        for _ in range(max(1, n_eval)):
            g, v = self._run_episode(training=False)
            returns.append(g)
            violations.append(v)
        return float(np.mean(returns)), float(np.mean(violations))

    def _prefill_replay_buffer(self):
        # Paper-style offline prefill: collect random-policy transitions for warmup epochs.
        warmup_returns = []
        warmup_violations = []
        prev_eps = self.epsilon
        self.epsilon = 1.0
        for _ in range(self.replay_warmup_epochs):
            g, v = self._run_episode(training=True)
            warmup_returns.append(g)
            warmup_violations.append(v)
        self.epsilon = prev_eps
        return warmup_returns, warmup_violations

    def train(self, episodes):
        episodes = int(episodes)
        episode_returns_local = []
        episode_violation_rates_local = []

        warmup_returns, warmup_violations = self._prefill_replay_buffer()
        episode_returns_local.extend(warmup_returns)
        episode_violation_rates_local.extend(warmup_violations)

        train_epochs = max(0, episodes - self.replay_warmup_epochs)
        for epoch in range(train_epochs):
            g, v = self._run_episode(training=True)
            episode_returns_local.append(g)
            episode_violation_rates_local.append(v)

            # Linear decay from epsilon_start to epsilon_end
            if train_epochs > 0:
                self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * ((epoch + 1) / train_epochs)
            else:
                self.epsilon = self.epsilon_end

            if (epoch + 1) % self.target_update_epochs == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            if self.eval_interval_epochs > 0 and (epoch + 1) % self.eval_interval_epochs == 0:
                eval_ret, eval_viol = self._evaluate_greedy()
                self.eval_trace.append({
                    "epoch": int(epoch + 1),
                    "avg_return": eval_ret,
                    "avg_violation": eval_viol,
                })

        return self.q_table, episode_returns_local, episode_violation_rates_local, self.epsilon


class CausalQLearningAgent(QLearningAgent):
    def __init__(self, env, causal_graph_adj=None, causal_graph_labels=None, graph_to_env_col=None,
                 target_label=None, parent_weight_threshold=1e-2, action_cols=None, latency_col="latency_col", **kwargs):
        if action_cols is None:
            action_cols = ["action_col"]
        self.causal_graph_adj = None if causal_graph_adj is None else np.asarray(causal_graph_adj, dtype=float)
        self.causal_graph_labels = [] if causal_graph_labels is None else list(causal_graph_labels)
        self.graph_to_env_col = {} if graph_to_env_col is None else dict(graph_to_env_col)
        self.target_label = None if target_label is None else str(target_label)
        self.parent_weight_threshold = float(parent_weight_threshold)

        self.parent_labels = []
        if (
            self.causal_graph_adj is not None
            and len(self.causal_graph_labels) > 0
            and self.target_label in self.causal_graph_labels
        ):
            target_idx = self.causal_graph_labels.index(self.target_label)
            self.parent_labels = [
                self.causal_graph_labels[j]
                for j in range(len(self.causal_graph_labels))
                if abs(self.causal_graph_adj[target_idx, j]) > self.parent_weight_threshold
            ]

        self.causal_state_cols = [
            self.graph_to_env_col[label]
            for label in self.parent_labels
            if label in self.graph_to_env_col and self.graph_to_env_col[label] in env.state_cols
        ]

        if not self.causal_state_cols:
            transition_candidates = [c for c in env.state_cols if c not in set(action_cols) | {latency_col}]
            self.causal_state_cols = list(action_cols)
            if transition_candidates:
                self.causal_state_cols.append(transition_candidates[0])

        # env state vector is [normalized state_cols..., extra_3_features].
        self.causal_state_indices = [env.state_cols.index(c) for c in self.causal_state_cols]

        super().__init__(env, feature_indices=self.causal_state_indices, **kwargs)


class QTablePolicyAdapter:
    def __init__(self, env, q_table):
        self.env = env
        self.q_table = q_table

    def greedy_action(self, state):
        return self.env.greedy_action(state, self.q_table)


def train_q_learning(env, episodes=1000, alpha=0.001, gamma=0.90,
                     epsilon_start=1.0, epsilon_end=0.0, epsilon_decay=1.0):
    agent = QLearningAgent(
        env,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
    )
    agent.train(episodes=episodes)
    return agent

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import random

class TabularQLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.90, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, decimals=2, feature_indices=None):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.decimals = decimals
        self.n_actions = len(env.actions)
        self.feature_indices = feature_indices

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))

    def _state_view(self, state):
        if isinstance(state, tuple) or isinstance(state, list):
            if self.feature_indices is not None:
                return tuple(state[i] for i in self.feature_indices)
            return tuple(state)
            
        arr = np.asarray(state, dtype=np.float32)
        if self.feature_indices is not None:
            arr = arr[self.feature_indices]
        return tuple(np.round(arr, decimals=self.decimals).tolist())

    def _select_action(self, state_key, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_table[state_key]))

    def greedy_action(self, state):
        state_key = self._state_view(state)
        return self._select_action(state_key, training=False)

    def train(self, episodes):
        returns = []
        violations = []

        for ep in range(episodes):
            state = self.env.reset()
            state_key = self._state_view(state)
            done = False
            G = 0.0
            ep_violations = 0
            ep_steps = 0

            while not done:
                a_idx = self._select_action(state_key, training=True)
                next_state, reward, done, info = self.env.step(a_idx)
                next_state_key = self._state_view(next_state)

                G += reward
                ep_violations += int(info.get("violation", 0))

                # Update Q-table
                best_next_q = np.max(self.q_table[next_state_key])
                td_target = reward + self.gamma * best_next_q * (1.0 - float(done))
                self.q_table[state_key][a_idx] += self.alpha * (td_target - self.q_table[state_key][a_idx])

                state_key = next_state_key
                ep_steps += 1

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            returns.append(G)
            violations.append(ep_violations / max(1, ep_steps))

        return self.q_table, returns, violations, self.epsilon


class CausalTabularQLearningAgent(TabularQLearningAgent):
    def __init__(self, env, causal_graph_adj=None, causal_graph_labels=None, graph_to_env_col=None,
                 target_label=None, parent_weight_threshold=1e-2, action_cols=None, latency_col="latency_col", **kwargs):
        if action_cols is None:
            action_cols = ["action_col"]
            
        self.causal_graph_adj = None if causal_graph_adj is None else np.asarray(causal_graph_adj, dtype=float)
        self.causal_graph_labels = [] if causal_graph_labels is None else list(causal_graph_labels)
        self.graph_to_env_col = {} if graph_to_env_col is None else dict(graph_to_env_col)
        self.target_label = target_label
        self.parent_weight_threshold = float(parent_weight_threshold)

        self.parent_labels = []
        if (self.causal_graph_adj is not None and
            len(self.causal_graph_labels) > 0 and
            self.target_label in self.causal_graph_labels):
            
            target_idx = self.causal_graph_labels.index(self.target_label)
            self.parent_labels = [
                self.causal_graph_labels[j] for j in range(len(self.causal_graph_labels))
                if abs(self.causal_graph_adj[target_idx, j]) > self.parent_weight_threshold
            ]

        self.causal_state_cols = [
            self.graph_to_env_col[label] for label in self.parent_labels
            if label in self.graph_to_env_col and self.graph_to_env_col[label] in env.state_cols
        ]

        if not self.causal_state_cols:
            transition_candidates = [c for c in env.state_cols if c not in set(action_cols) | {latency_col}]
            self.causal_state_cols = list(action_cols)
            if transition_candidates:
                self.causal_state_cols.append(transition_candidates[0])

        self.causal_state_indices = [env.state_cols.index(c) for c in self.causal_state_cols]

        super().__init__(env, feature_indices=self.causal_state_indices, **kwargs)
