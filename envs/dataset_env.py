import numpy as np

class DatasetEnv:
    def __init__(
        self,
        dataset_info,
        artifacts,
        sla_latency_ms,
        max_steps,
        action_cols,
        latency_col,
        r_max=100.0,
        w_lat=1.0/3.0,
        w_res=2.0/3.0,
        lat_below_ref_decay=0.06,
        lat_above_ref_decay=10.0,
        lat_shape_gain=1.0,
        invalid_or_slo_penalty=-100.0,
        is_discrete=False,
    ):
        self.df = dataset_info["df"]
        self.state_cols = dataset_info["state_cols"]
        self.actions = dataset_info["actions"]
        self.action_to_idx = dataset_info["action_to_idx"]
        self.idx_to_action = dataset_info["idx_to_action"]

        self.action_cols = action_cols
        self.latency_col = latency_col

        self.state_raw = artifacts["state_matrix"]
        self.state_norm = artifacts["state_matrix_norm"]
        self.discretize_row = artifacts["discretize_row"]

        self.sla_latency_ms = float(sla_latency_ms)
        self.max_steps = int(max_steps)

        self.r_max = float(r_max)
        self.w_lat = float(w_lat)
        self.w_res = float(w_res)
        self.lat_below_ref_decay = float(lat_below_ref_decay)
        self.lat_above_ref_decay = float(lat_above_ref_decay)
        self.lat_shape_gain = float(lat_shape_gain)
        self.invalid_or_slo_penalty = float(invalid_or_slo_penalty)
        self.is_discrete = is_discrete

        self.transition_cols = [c for c in self.state_cols if c not in set(self.action_cols) | {self.latency_col}]
        self.transition_idx = [self.state_cols.index(c) for c in self.transition_cols]

        self.indices_by_action = {}
        for a in self.actions:
             mask = np.ones(len(self.df), dtype=bool)
             for i, col in enumerate(self.action_cols):
                 mask &= (self.df[col].values == a[i])
             self.indices_by_action[a] = np.where(mask)[0]

        self.observed_actions = list(set(tuple(x) for x in self.df[self.action_cols].astype(int).to_numpy()))

        self.max_action_sum = max([sum(a) for a in self.actions]) if self.actions else 1

        self.cur_idx = None
        self.t = 0

        self.state_dim = int(self.state_norm.shape[1] + 2 + len(self.action_cols))

    def _reward(self, latency_ms, action_tuple, action_invalid=False):
        z = float(latency_ms) / max(1e-8, float(self.sla_latency_ms))
        if z <= 1.0:
            r_lat = self.r_max * np.exp(-self.lat_below_ref_decay * self.lat_shape_gain * (1.0 - z) ** 2)
        else:
            r_lat = self.r_max * np.exp(-self.lat_above_ref_decay * self.lat_shape_gain * (1.0 - z) ** 2)

        s_max = max(1.0, float(self.max_action_sum))
        total_replicas = float(sum(action_tuple))
        r_res = self.r_max * ((s_max - total_replicas) / s_max)

        slo_unreached = bool(z > 1.0)
        if action_invalid or slo_unreached:
            return self.invalid_or_slo_penalty

        return float(self.w_lat * r_lat + self.w_res * r_res)

    def _nearest_candidates(self, action, k=1):
        current_norm = self.state_norm[self.cur_idx, self.transition_idx]

        candidates = self.indices_by_action.get(action, np.array([], dtype=int))
        used_action = action
        if len(candidates) == 0 and self.observed_actions:
            def dist(a1, a2):
                return sum((v1 - v2)**2 for v1, v2 in zip(a1, a2))
            used_action = min(self.observed_actions, key=lambda a: dist(a, action))
            candidates = self.indices_by_action.get(used_action, np.array([], dtype=int))

        if len(candidates) == 0:
            candidates = np.arange(len(self.df), dtype=int)
            used_action = tuple(int(self.df.iloc[int(candidates[0])][c]) for c in self.action_cols)

        cand_norm = self.state_norm[candidates][:, self.transition_idx]
        d2 = ((cand_norm - current_norm) ** 2).sum(axis=1)
        order = np.argsort(d2)
        k = max(1, min(k, len(order)))
        topk = candidates[order[:k]]
        return topk, used_action

    def _state_vector(self, idx):
        if self.is_discrete:
            return self.discretize_row(self.state_raw[idx])
        base = self.state_norm[idx].astype(np.float32)
        latency = float(self.df.iloc[idx][self.latency_col])
        replicas = [float(self.df.iloc[idx][c]) for c in self.action_cols]
        extras = [latency / max(1.0, self.sla_latency_ms)] + [r / max(1.0, float(self.max_action_sum)) for r in replicas] + [1.0]
        return np.concatenate([base, np.array(extras, dtype=np.float32)]).astype(np.float32)

    def greedy_action(self, state, q_table):
        key = tuple(np.asarray(state, dtype=float).round(4).tolist())
        if key in q_table:
            return int(np.argmax(q_table[key]))
        return np.random.randint(len(self.actions))

    def reset(self):
        self.cur_idx = np.random.randint(0, len(self.df))
        self.t = 0
        return self._state_vector(self.cur_idx)

    def step(self, action_idx):
        action = self.idx_to_action[action_idx]
        topk, used_action = self._nearest_candidates(action, k=1)
        next_idx = int(topk[0])

        self.cur_idx = next_idx
        self.t += 1

        latency = float(self.df.iloc[next_idx][self.latency_col])
        action_invalid = bool(used_action != action)
        reward = self._reward(latency, used_action, action_invalid=action_invalid)
        next_state = self._state_vector(next_idx)

        done = self.t >= self.max_steps
        info = {
            "latency": latency,
            "replicas": used_action,
            "violation": float(latency > self.sla_latency_ms),
            "action_invalid": float(action_invalid),
        }
        return next_state, reward, done, info
