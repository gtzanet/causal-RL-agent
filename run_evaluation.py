import os
import argparse
import yaml
import json
import uuid
import datetime
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Agent/Env imports
from envs.dataset_env import DatasetEnv
from causality.causal_discovery import CausalDiscovery
from agents.agents import (
    DQNNetwork,
    ReplayBuffer,
    QLearningAgent,
    CausalQLearningAgent,
    QTablePolicyAdapter,
    train_q_learning,
    TabularQLearningAgent,
    CausalTabularQLearningAgent
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Evaluation Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    return parser.parse_args()

def setup_results_dir(config_path, config_data):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save a copy of the config
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.dump(config_data, f)
        
    return results_dir

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Dataset loading helpers
def _validate_required_columns(df_local, required_cols, path_label):
    missing = [c for c in required_cols if c not in df_local.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path_label}: {missing}")

def _resolve_state_columns(df_local, state_col_prefix, state_cols_override, action_cols, latency_col):
    if state_cols_override is not None:
        state_cols_local = list(dict.fromkeys(state_cols_override))
    elif state_col_prefix is not None:
        state_cols_local = [c for c in df_local.columns if c.startswith(state_col_prefix)]
    else:
        state_cols_local = [c for c in df_local.columns if c not in set(action_cols) | {latency_col}]

    if not state_cols_local:
        raise ValueError("State space is empty.")

    for ac in action_cols:
        if ac not in state_cols_local:
            state_cols_local.append(ac)
    if latency_col not in state_cols_local:
        state_cols_local.append(latency_col)

    return state_cols_local

def _normalize_actions(df_actions, explicit_action_space=None):
    if explicit_action_space is None:
        return sorted(list(set(tuple(x) for x in df_actions.to_numpy())))
    return sorted(list(set(tuple(int(v) for v in a) for a in explicit_action_space)))

def load_od_dataset(path, config):
    data = pd.read_csv(path).dropna().reset_index(drop=True)
    features = config["features"]
    
    keep_columns = features.get("keep_columns")
    action_cols = features["action_columns"]
    latency_col = features["latency_column"]
    
    if keep_columns is not None:
        keep = list(dict.fromkeys(list(keep_columns) + action_cols + [latency_col]))
        _validate_required_columns(data, keep, path)
        data = data[keep].copy()

    _validate_required_columns(data, action_cols + [latency_col], path)
    for ac in action_cols:
        data[ac] = data[ac].round().astype(int)

    state_cols_local = _resolve_state_columns(
        data, 
        features.get("state_col_prefix"), 
        features.get("state_space_cols"), 
        action_cols, 
        latency_col
    )
    actions_local = _normalize_actions(data[action_cols], explicit_action_space=features.get("action_space"))

    return {"path": path, "df": data, "state_cols": state_cols_local, "actions": actions_local}

def dataset_info_from_df(df, path_label, config):
    data = df.copy().dropna().reset_index(drop=True)
    features = config["features"]
    
    keep_columns = features.get("keep_columns")
    action_cols = features["action_columns"]
    latency_col = features["latency_column"]

    if keep_columns is not None:
        keep = list(dict.fromkeys(list(keep_columns) + action_cols + [latency_col]))
        _validate_required_columns(data, keep, path_label)
        data = data[keep].copy()

    _validate_required_columns(data, action_cols + [latency_col], path_label)
    for ac in action_cols:
        data[ac] = data[ac].round().astype(int)

    state_cols_local = _resolve_state_columns(
        data, 
        features.get("state_col_prefix"), 
        features.get("state_space_cols"), 
        action_cols, 
        latency_col
    )
    actions_local = _normalize_actions(data[action_cols], explicit_action_space=features.get("action_space"))

    return {"path": path_label, "df": data, "state_cols": state_cols_local, "actions": actions_local}

def set_action_space(dataset_info, actions_space):
    dataset_info["actions"] = sorted(list(set(tuple(int(v) for v in a) for a in actions_space)))
    dataset_info["action_to_idx"] = {a: i for i, a in enumerate(dataset_info["actions"])}
    dataset_info["idx_to_action"] = {i: a for a, i in dataset_info["action_to_idx"].items()}

def build_dataset_artifacts(dataset_info, n_bins):
    df_local = dataset_info["df"]
    state_cols_local = dataset_info["state_cols"]

    state_matrix_local = df_local[state_cols_local].to_numpy(dtype=float)
    mu_local = state_matrix_local.mean(axis=0)
    sigma_local = state_matrix_local.std(axis=0) + 1e-8
    state_matrix_norm_local = (state_matrix_local - mu_local) / sigma_local

    bin_edges_local = {}
    for c in state_cols_local:
        q = np.linspace(0, 1, n_bins + 1)
        e = np.quantile(df_local[c].to_numpy(dtype=float), q)
        e = np.unique(e)
        if len(e) < 3:
            v = df_local[c].to_numpy(dtype=float)
            e = np.array([v.min() - 1e-6, v.max() + 1e-6])
        bin_edges_local[c] = e

    def discretize_row_local(row_values):
        s = []
        for i, c in enumerate(state_cols_local):
            e = bin_edges_local[c]
            b = np.digitize(row_values[i], e[1:-1], right=False)
            s.append(int(b))
        return tuple(s)

    return {
        "state_matrix": state_matrix_local,
        "state_matrix_norm": state_matrix_norm_local,
        "mu": mu_local,
        "sigma": sigma_local,
        "bin_edges": bin_edges_local,
        "discretize_row": discretize_row_local,
    }

def make_env(dataset_info, artifacts, config, sla_latency_ms=None):
    if sla_latency_ms is None:
        sla_latency_ms = float(config["reward_function"]["sla_latency_ms"])
        
    return DatasetEnv(
        dataset_info=dataset_info,
        artifacts=artifacts,
        sla_latency_ms=sla_latency_ms,
        max_steps=config["split"]["max_steps_per_episode"],
        action_cols=config["features"]["action_columns"],
        latency_col=config["features"]["latency_column"],
        r_max=config["reward_function"]["r_max"],
        w_lat=config["reward_function"]["w_lat"],
        w_res=config["reward_function"]["w_res"],
        lat_below_ref_decay=config["reward_function"]["lat_below_ref_decay"],
        lat_above_ref_decay=config["reward_function"]["lat_above_ref_decay"],
        lat_shape_gain=config["reward_function"]["lat_shape_gain"],
        invalid_or_slo_penalty=config["reward_function"]["invalid_or_slo_penalty"],
        is_discrete=config["agents"]["use_tabular_agent"],
    )

def _get_agent_kwargs(config, is_causal=False):
    kwargs = {}
    if config["agents"]["use_tabular_agent"]:
        kwargs.update({
            "alpha": config["training"]["alpha"],
            "gamma": config["training"]["gamma"],
            "epsilon_start": config["training"]["epsilon_start"],
            "epsilon_end": config["training"]["epsilon_end"],
            "epsilon_decay": config["training"]["epsilon_decay"]
        })
    if is_causal:
        kwargs.update({
            "causal_graph_adj": config["causal"].get("adj"),
            "causal_graph_labels": config["causal"].get("labels"),
            "graph_to_env_col": config["causal"].get("to_env_col"),
            "target_label": config["causal"].get("target_label"),
            "parent_weight_threshold": config["causal"].get("parent_weight_threshold"),
            "action_cols": config["features"]["action_columns"],
            "latency_col": config["features"]["latency_column"]
        })
    return kwargs

def evaluate_policy(env, policy, episodes=200):
    if hasattr(policy, "greedy_action"):
        policy_obj = policy
    else:
        policy_obj = QTablePolicyAdapter(env, policy)

    total_reward = 0.0
    latencies = []
    # `replicas` now stores tuples of actions
    replicas = []
    violations = 0
    steps = 0
    t1_list = []

    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a_idx = policy_obj.greedy_action(s)
            s, r, done, info = env.step(a_idx)
            total_reward += r
            latencies.append(info["latency"])
            replicas.append(info["replicas"]) # info["replicas"] is a tuple
            violations += int(info["violation"])
            t1_val = float(env.df.iloc[env.cur_idx]["service_frame_sampler_arrival_rate"])
            t1_list.append(t1_val)
            steps += 1

    avg_tuple = tuple(np.mean([r[i] for r in replicas]) for i in range(len(env.action_cols)))
    
    # Calculate t1 averages
    df_t1 = pd.DataFrame({"t1": t1_list})
    for i, col in enumerate(env.action_cols):
        df_t1[col] = [r[i] for r in replicas]
    t1_dict = df_t1.groupby("t1").mean().reset_index().to_dict(orient="list")

    return {
        "avg_reward": total_reward / steps,
        "avg_latency": float(np.mean(latencies)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "avg_replicas": float(np.mean([sum(r) for r in replicas])),
        "avg_replicas_tuple": str(tuple(round(x, 2) for x in avg_tuple)),
        "violation_rate": violations / steps,
        "total_steps": steps,
        "t1_replicas_dict": t1_dict,
    }

def multiset_intersection_left(ud_df, bd_df):
    common_cols = [c for c in ud_df.columns if c in bd_df.columns]
    ud = ud_df[common_cols].copy()
    bd = bd_df[common_cols].copy()

    ud["_sig"] = pd.util.hash_pandas_object(ud.round(5).astype(str), index=False)
    bd["_sig"] = pd.util.hash_pandas_object(bd.round(5).astype(str), index=False)

    ud["_occ"] = ud.groupby("_sig").cumcount()
    bd["_occ"] = bd.groupby("_sig").cumcount()

    ud["_key"] = ud["_sig"].astype(str) + "_" + ud["_occ"].astype(str)
    bd["_key"] = bd["_sig"].astype(str) + "_" + bd["_occ"].astype(str)
    
    mask = ud["_key"].isin(bd["_key"])
    return ud_df[mask].copy()


def multiset_difference_df(ud_df, bd_df):
    common_cols = [c for c in ud_df.columns if c in bd_df.columns]
    ud = ud_df[common_cols].copy()
    bd = bd_df[common_cols].copy()

    ud["_sig"] = pd.util.hash_pandas_object(ud.round(5).astype(str), index=False)
    bd["_sig"] = pd.util.hash_pandas_object(bd.round(5).astype(str), index=False)

    ud["_occ"] = ud.groupby("_sig").cumcount()
    bd["_occ"] = bd.groupby("_sig").cumcount()

    ud["_key"] = ud["_sig"].astype(str) + "_" + ud["_occ"].astype(str)
    bd["_key"] = bd["_sig"].astype(str) + "_" + bd["_occ"].astype(str)
    
    mask = ~ud["_key"].isin(bd["_key"])
    return ud_df[mask].copy()

def main():
    args = parse_args()
    config = load_config(args.config)
    results_dir = setup_results_dir(args.config, config)
    print(f"Results will be saved to: {results_dir}")

    # Set seeds
    np.random.seed(17)
    random.seed(17)
    torch.manual_seed(17)

    train_data_path = config["dataset"]["train_data_path"]
    eval_data_path = config["dataset"]["eval_data_path"]

    print("Loading datasets...")
    train_data = load_od_dataset(train_data_path, config)
    ud_full_data = load_od_dataset(eval_data_path, config)

    rng = np.random.default_rng(config["split"]["ud_split_seed"])
    n_samples = len(ud_full_data["df"])
    perm = rng.permutation(n_samples)
    split_idx = int(n_samples * config["split"]["ud_train_split_frac"])
    train_idx = perm[:split_idx]
    test_idx = perm[split_idx:]

    ud_train_data = dataset_info_from_df(ud_full_data["df"].iloc[train_idx], "eval_train_data", config)
    ud_test_data = dataset_info_from_df(ud_full_data["df"].iloc[test_idx], "eval_test_data", config)

    GLOBAL_ACTIONS = sorted(set(train_data["actions"]) | set(ud_full_data["actions"]))
    set_action_space(train_data, GLOBAL_ACTIONS)
    set_action_space(ud_train_data, GLOBAL_ACTIONS)
    set_action_space(ud_test_data, GLOBAL_ACTIONS)
    eval_data = ud_test_data

    n_bins = config["split"]["n_bins"]
    train_artifacts = build_dataset_artifacts(train_data, n_bins)
    ud_train_artifacts = build_dataset_artifacts(ud_train_data, n_bins)
    ud_test_artifacts = build_dataset_artifacts(ud_test_data, n_bins)
    eval_artifacts = ud_test_artifacts

    # Causal Discovery
    print("Running Causal Discovery...")
    USE_NOTEARS_CAUSAL_GRAPH = config["causal"].get("use_notears", True)
    NOTEARS_MAX_ITER = config["causal"].get("notears_max_iter", 100)
    NOTEARS_WEIGHT_THRESHOLD = config["causal"].get("notears_weight_threshold", 5.0)
    NOTEARS_FIXED_EDGES = config["causal"].get("notears_fixed_edges")
    CAUSAL_DISCOVERY_COLS = config["causal"].get("causal_discovery_cols", train_data["state_cols"])
    latency_col = config["features"]["latency_column"]

    if USE_NOTEARS_CAUSAL_GRAPH:
        discovery_cols = list(dict.fromkeys(CAUSAL_DISCOVERY_COLS))
        def discover_graph_for_df(df_source, name):
            actual_cols = [c for c in discovery_cols if c in df_source.columns]
            df_target = df_source[actual_cols].copy()
            try:
                cd = CausalDiscovery(method="notears")
                cd.fit(
                    df_target,
                    notears_w_threshold=NOTEARS_WEIGHT_THRESHOLD,
                    notears_max_iter=NOTEARS_MAX_ITER,
                    fixed_edges=NOTEARS_FIXED_EDGES,
                )
                labels = list(cd.labels_)
                adj = np.asarray(cd.adjacency_matrix_, dtype=float)
                mapping = {c: c for c in labels}
                target = latency_col if latency_col in labels else None
                return {"labels": labels, "adj": adj, "mapping": mapping, "target": target, "error": None}
            except Exception as exc:
                return {"labels": None, "adj": None, "mapping": None, "target": None, "error": str(exc)}

        bd_causal_graph = discover_graph_for_df(train_data["df"], "BD")
        ud_causal_graph = discover_graph_for_df(ud_train_data["df"], "UD")

        # Save causal matrices
        np.save(os.path.join(results_dir, "bd_adj.npy"), bd_causal_graph["adj"] if bd_causal_graph["adj"] is not None else np.array([]))
        np.save(os.path.join(results_dir, "ud_adj.npy"), ud_causal_graph["adj"] if ud_causal_graph["adj"] is not None else np.array([]))
        with open(os.path.join(results_dir, "causal_labels.json"), "w") as f:
            json.dump({
                "bd_labels": bd_causal_graph["labels"],
                "bd_target": bd_causal_graph["target"],
                "ud_labels": ud_causal_graph["labels"],
                "ud_target": ud_causal_graph["target"]
            }, f)

        if bd_causal_graph["error"] is None:
            config["causal"]["labels"] = bd_causal_graph["labels"]
            config["causal"]["adj"] = bd_causal_graph["adj"]
            config["causal"]["to_env_col"] = bd_causal_graph["mapping"]
            config["causal"]["target_label"] = bd_causal_graph["target"]
    else:
        print("Skipping automatic causal discovery")
        bd_causal_graph = {}
        ud_causal_graph = {}

    AgentClass = TabularQLearningAgent if config["agents"]["use_tabular_agent"] else QLearningAgent
    CausalAgentClass = CausalTabularQLearningAgent if config["agents"]["use_tabular_agent"] else CausalQLearningAgent

    print("Training standard agents...")
    biased_train_env = make_env(train_data, train_artifacts, config)
    unbiased_train_env = make_env(ud_train_data, ud_train_artifacts, config)
    unbiased_eval_env = make_env(ud_test_data, ud_test_artifacts, config)
    causal_train_env = make_env(train_data, train_artifacts, config)

    biased_agent = AgentClass(biased_train_env, **_get_agent_kwargs(config))
    _, episode_returns_biased, _, _ = biased_agent.train(episodes=config["training"]["episodes"])

    unbiased_agent = AgentClass(unbiased_train_env, **_get_agent_kwargs(config))
    _, episode_returns_unbiased, _, _ = unbiased_agent.train(episodes=config["training"]["episodes"])

    causal_biased_agent = CausalAgentClass(causal_train_env, **_get_agent_kwargs(config, is_causal=True))
    _, episode_returns_causal_biased, _, _ = causal_biased_agent.train(episodes=config["training"]["episodes"])

    # Save training curves
    training_curves = pd.DataFrame({
        "biased": episode_returns_biased,
        "unbiased": episode_returns_unbiased,
        "causal_biased": episode_returns_causal_biased
    })
    training_curves.to_csv(os.path.join(results_dir, "training_curves.csv"), index=False)

    print("Evaluating baseline agents...")
    biased_eval = evaluate_policy(unbiased_eval_env, biased_agent, episodes=300)
    unbiased_eval = evaluate_policy(unbiased_eval_env, unbiased_agent, episodes=300)
    causal_eval = evaluate_policy(unbiased_eval_env, causal_biased_agent, episodes=300)

    comparison_df = pd.DataFrame([
        {"agent": "biased-trained", **{k:v for k,v in biased_eval.items() if k != "t1_replicas_dict"}},
        {"agent": "unbiased-trained", **{k:v for k,v in unbiased_eval.items() if k != "t1_replicas_dict"}},
        {"agent": "causal-biased-trained", **{k:v for k,v in causal_eval.items() if k != "t1_replicas_dict"}}
    ])
    comparison_df.to_csv(os.path.join(results_dir, "baseline_comparison.csv"), index=False)

    # Save t1 replicas
    t1_rows = []
    for agent_name, eval_dict in [("biased-trained", biased_eval), 
                                  ("unbiased-trained", unbiased_eval), 
                                  ("causal-biased-trained", causal_eval)]:
        d = eval_dict["t1_replicas_dict"]
        for i in range(len(d["t1"])):
            row = {"agent": agent_name, "t1": d["t1"][i]}
            for col in unbiased_eval_env.action_cols:
                row[col] = d[col][i]
            t1_rows.append(row)
    pd.DataFrame(t1_rows).to_csv(os.path.join(results_dir, "t1_replicas.csv"), index=False)

    print(comparison_df)

    # Calculate optimal SLA
    print("Auto-relaxing SLA...")
    def train_and_eval_for_sla(train_dataset, train_ds_artifacts, eval_dataset, eval_ds_artifacts, sla_value, train_episodes=1500, eval_episodes=200):
        local_train_env = make_env(train_dataset, train_ds_artifacts, config, sla_latency_ms=float(sla_value))
        local_eval_env = make_env(eval_dataset, eval_ds_artifacts, config, sla_latency_ms=float(sla_value))
        local_agent = AgentClass(local_train_env, **_get_agent_kwargs(config))
        local_agent.train(episodes=train_episodes)
        m_train = evaluate_policy(local_train_env, local_agent, episodes=eval_episodes)
        m_eval = evaluate_policy(local_eval_env, local_agent, episodes=eval_episodes)
        return m_train, m_eval

    candidate_slas = list(range(150, 501, 25))
    search_rows = []
    for sla in candidate_slas:
        m_train, m_eval = train_and_eval_for_sla(
            train_data, train_artifacts, eval_data, eval_artifacts, 
            sla_value=sla, train_episodes=1200, eval_episodes=200
        )
        search_rows.append({
            "sla": sla,
            "train_violation_rate": m_train["violation_rate"],
            "eval_violation_rate": m_eval["violation_rate"],
            "eval_avg_replicas": m_eval["avg_replicas"],
            "eval_avg_latency": m_eval["avg_latency"],
        })
    sla_search_df = pd.DataFrame(search_rows)
    sla_search_df.to_csv(os.path.join(results_dir, "sla_search.csv"), index=False)

    TARGET_VIOLATION_RATE = config["threshold_experiment"].get("target_violation_rate", 0.3)
    feasible = sla_search_df[sla_search_df["eval_violation_rate"] < TARGET_VIOLATION_RATE]
    if len(feasible) > 0:
        chosen_sla = float(feasible.iloc[0]["sla"])
    else:
        chosen_sla = float(sla_search_df.iloc[sla_search_df["eval_violation_rate"].argmin()]["sla"])
    
    SLA_LATENCY_MS = chosen_sla
    print(f"Chosen SLA: {SLA_LATENCY_MS}")
    config["reward_function"]["sla_latency_ms"] = SLA_LATENCY_MS
    
    # Save the updated config with chosen SLA
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # RD Experiment
    print("Running RD Experiment...")
    bd_source_df = train_data["df"].copy()
    ud_train_df = ud_train_data["df"].copy().reset_index(drop=True)
    base_bd_df = multiset_intersection_left(bd_source_df, ud_train_df)
    ud_minus_bd_df = multiset_difference_df(ud_train_df, base_bd_df)

    ADD_PER_RUN = max(1, len(ud_minus_bd_df) // 10)
    N_REPEATS = config["rd_experiment"]["n_repeats"]
    RD_TRAIN_EPISODES = config["rd_experiment"]["rd_train_episodes"]
    RD_EVAL_EPISODES = config["rd_experiment"]["rd_eval_episodes"]
    POINT_SEED = config["rd_experiment"]["point_seed"]

    rd_t1_rows = []

    def train_unbiased_baseline(train_episodes):
        unbiased_env_local = make_env(ud_train_data, ud_train_artifacts, config, sla_latency_ms=SLA_LATENCY_MS)
        unbiased_agent_local = AgentClass(unbiased_env_local, **_get_agent_kwargs(config))
        unbiased_agent_local.train(episodes=train_episodes)
        return unbiased_agent_local

    unbiased_eval_env_local = make_env(ud_test_data, ud_test_artifacts, config, sla_latency_ms=SLA_LATENCY_MS)
    unbiased_violations = []
    unbiased_replicas = []
    
    for rep_i in range(N_REPEATS):
        unbiased_baseline_agent = train_unbiased_baseline(train_episodes=RD_TRAIN_EPISODES)
        unbiased_metrics = evaluate_policy(unbiased_eval_env_local, unbiased_baseline_agent, episodes=RD_EVAL_EPISODES)
        d_unb = unbiased_metrics.pop("t1_replicas_dict", None)
        if d_unb is not None:
            for i in range(len(d_unb["t1"])):
                row = {"agent": "unbiased-ud-q baseline", "repeat": rep_i, "t1": d_unb["t1"][i]}
                for col in unbiased_eval_env_local.action_cols: row[col] = d_unb[col][i]
                rd_t1_rows.append(row)
        unbiased_violations.append(unbiased_metrics["violation_rate"])
        unbiased_replicas.append(unbiased_metrics["avg_replicas"])
        
    unbiased_violation_baseline = float(np.mean(unbiased_violations))
    unbiased_replicas_baseline = float(np.mean(unbiased_replicas))

    with open(os.path.join(results_dir, "rd_baselines.json"), "w") as f:
        json.dump({
            "unbiased_violation_baseline": unbiased_violation_baseline,
            "unbiased_replicas_baseline": unbiased_replicas_baseline
        }, f)

    rd_rows = []
    max_added = len(ud_minus_bd_df)
    added_points_grid = sorted(list(set(int(x) for x in np.linspace(0, max_added, 11))))

    for rep in range(N_REPEATS):
        rng_local = np.random.default_rng(POINT_SEED + rep)
        pool = ud_minus_bd_df.iloc[rng_local.permutation(len(ud_minus_bd_df))].reset_index(drop=True)

        for added_points in added_points_grid:
            added_df = pool.iloc[:added_points].copy()
            rd_df = pd.concat([base_bd_df, added_df], ignore_index=True)
            if len(rd_df) == 0:
                continue
            rd_df = rd_df.sample(frac=1.0, random_state=rep).reset_index(drop=True)

            rd_info = dataset_info_from_df(rd_df, f"RD_seed{POINT_SEED}_rep{rep}_add{added_points}", config)
            set_action_space(rd_info, GLOBAL_ACTIONS)
            rd_artifacts_local = build_dataset_artifacts(rd_info, n_bins)
            rd_train_env = make_env(rd_info, rd_artifacts_local, config, sla_latency_ms=SLA_LATENCY_MS)
            eval_env_local2 = make_env(ud_test_data, ud_test_artifacts, config, sla_latency_ms=SLA_LATENCY_MS)

            rd_q_agent = AgentClass(rd_train_env, **_get_agent_kwargs(config))
            rd_q_agent.train(episodes=RD_TRAIN_EPISODES)

            rd_causal_agent = CausalAgentClass(rd_train_env, **_get_agent_kwargs(config, is_causal=True))
            rd_causal_agent.train(episodes=RD_TRAIN_EPISODES)

            metrics_rd_q = evaluate_policy(eval_env_local2, rd_q_agent, episodes=RD_EVAL_EPISODES)
            metrics_rd_causal = evaluate_policy(eval_env_local2, rd_causal_agent, episodes=RD_EVAL_EPISODES)
            
            d_q = metrics_rd_q.pop("t1_replicas_dict", None)
            d_causal = metrics_rd_causal.pop("t1_replicas_dict", None)
            
            # Save t1 replicas for RD
            if rd_t1_rows is not None and added_points == max_added: # only save for the final model
                for i in range(len(d_q["t1"])):
                    row = {"agent": "rd-q", "repeat": rep, "t1": d_q["t1"][i]}
                    for col in eval_env_local2.action_cols: row[col] = d_q[col][i]
                    rd_t1_rows.append(row)
                for i in range(len(d_causal["t1"])):
                    row = {"agent": "rd-causal-q", "repeat": rep, "t1": d_causal["t1"][i]}
                    for col in eval_env_local2.action_cols: row[col] = d_causal[col][i]
                    rd_t1_rows.append(row)

            rd_rows.extend([
                {"point_seed": int(POINT_SEED), "repeat": rep, "added_points": added_points, "rd_size": len(rd_df), "agent": "rd-q", **metrics_rd_q},
                {"point_seed": int(POINT_SEED), "repeat": rep, "added_points": added_points, "rd_size": len(rd_df), "agent": "rd-causal-q", **metrics_rd_causal},
            ])

    rd_results_df = pd.DataFrame(rd_rows)
    rd_results_df.to_csv(os.path.join(results_dir, "rd_results.csv"), index=False)
    
    if len(rd_t1_rows) > 0:
        pd.DataFrame(rd_t1_rows).to_csv(os.path.join(results_dir, "rd_t1_replicas.csv"), index=False)

    rd_summary = rd_results_df.groupby(["added_points", "agent"], as_index=False).agg(
        violation_rate_mean=("violation_rate", "mean"),
        violation_rate_std=("violation_rate", "std"),
        avg_replicas_mean=("avg_replicas", "mean"),
        avg_replicas_std=("avg_replicas", "std"),
        avg_reward_mean=("avg_reward", "mean"),
    ).sort_values(["agent", "added_points"])
    rd_summary.to_csv(os.path.join(results_dir, "rd_summary.csv"), index=False)

    # Threshold experiment
    print("Running Threshold Experiment...")
    RD_ADDED_POINTS = max(1, int(len(ud_minus_bd_df) * config["threshold_experiment"]["rd_added_points_ratio"]))
    RD60_POINT_SEED = config["threshold_experiment"]["rd60_point_seed"]
    THRESHOLD_VALUES = config["threshold_experiment"]["threshold_values"]
    THRESHOLD_TRAIN_EPISODES = config["threshold_experiment"]["threshold_train_episodes"]
    THRESHOLD_EVAL_EPISODES = config["threshold_experiment"]["threshold_eval_episodes"]
    THRESHOLD_N_REPEATS = config["threshold_experiment"]["n_repeats"]

    n_add = min(RD_ADDED_POINTS, len(ud_minus_bd_df))
    results = []

    for rep in range(THRESHOLD_N_REPEATS):
        rep_seed = RD60_POINT_SEED + rep
        rng_rd60 = np.random.default_rng(rep_seed)
        added_df_60 = ud_minus_bd_df.iloc[rng_rd60.permutation(len(ud_minus_bd_df))[:n_add]].copy()
        rd60_df = pd.concat([base_bd_df, added_df_60], ignore_index=True)
        if len(rd60_df) > 0:
            rd60_df = rd60_df.sample(frac=1.0, random_state=rep_seed).reset_index(drop=True)
        rd60_info = dataset_info_from_df(rd60_df, f"RD_plus_{n_add}_points_rep_{rep}", config)
        set_action_space(rd60_info, GLOBAL_ACTIONS)
        rd60_artifacts = build_dataset_artifacts(rd60_info, n_bins)

        for sla in THRESHOLD_VALUES:
            # Unbiased baseline
            unbiased_train_env_loc = make_env(ud_train_data, ud_train_artifacts, config, sla_latency_ms=float(sla))
            unbiased_agent_loc = AgentClass(unbiased_train_env_loc, **_get_agent_kwargs(config))
            unbiased_agent_loc.train(episodes=THRESHOLD_TRAIN_EPISODES)

            # RD-Q
            rd_q_train_env_loc = make_env(rd60_info, rd60_artifacts, config, sla_latency_ms=float(sla))
            rd_q_agent_loc = AgentClass(rd_q_train_env_loc, **_get_agent_kwargs(config))
            rd_q_agent_loc.train(episodes=THRESHOLD_TRAIN_EPISODES)

            # RD-Causal
            rd_causal_train_env_loc = make_env(rd60_info, rd60_artifacts, config, sla_latency_ms=float(sla))
            rd_causal_agent_loc = CausalAgentClass(rd_causal_train_env_loc, **_get_agent_kwargs(config, is_causal=True))
            rd_causal_agent_loc.train(episodes=THRESHOLD_TRAIN_EPISODES)

            eval_env_local_th = make_env(ud_test_data, ud_test_artifacts, config, sla_latency_ms=float(sla))
            m_unbiased = evaluate_policy(eval_env_local_th, unbiased_agent_loc, episodes=THRESHOLD_EVAL_EPISODES)
            m_rd_q = evaluate_policy(eval_env_local_th, rd_q_agent_loc, episodes=THRESHOLD_EVAL_EPISODES)
            m_rd_causal = evaluate_policy(eval_env_local_th, rd_causal_agent_loc, episodes=THRESHOLD_EVAL_EPISODES)
            
            m_unbiased.pop("t1_replicas_dict", None)
            m_rd_q.pop("t1_replicas_dict", None)
            m_rd_causal.pop("t1_replicas_dict", None)

            results.extend([
                {"repeat": rep, "model": "unbiased-ud-q", "sla_latency_ms": float(sla), **m_unbiased},
                {"repeat": rep, "model": f"rd-q (+{n_add})", "sla_latency_ms": float(sla), **m_rd_q},
                {"repeat": rep, "model": f"rd-causal-q (+{n_add})", "sla_latency_ms": float(sla), **m_rd_causal},
            ])
            pd.DataFrame(results).to_csv(os.path.join(results_dir, "threshold_results.tmp.csv"), index=False) # Checkpoint

    rd60_vs_sla_df = pd.DataFrame(results)
    rd60_vs_sla_df.to_csv(os.path.join(results_dir, "threshold_results.csv"), index=False)
    if os.path.exists(os.path.join(results_dir, "threshold_results.tmp.csv")):
        os.remove(os.path.join(results_dir, "threshold_results.tmp.csv"))

    rd60_summary = rd60_vs_sla_df.groupby(["sla_latency_ms", "model"], as_index=False).agg(
        violation_rate_mean=("violation_rate", "mean"),
        violation_rate_min=("violation_rate", "min"),
        violation_rate_max=("violation_rate", "max"),
        avg_replicas_mean=("avg_replicas", "mean"),
        avg_replicas_min=("avg_replicas", "min"),
        avg_replicas_max=("avg_replicas", "max"),
    ).sort_values(["model", "sla_latency_ms"])
    rd60_summary.to_csv(os.path.join(results_dir, "threshold_summary.csv"), index=False)

    print(f"Finished! Everything saved to {results_dir}")

if __name__ == "__main__":
    main()
