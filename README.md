# Causal RL Agent

Reinforcement-learning agents for autoscaling microservices, where the agent uses causal discovery to correct for bias in the data it learns from.

Monitoring and autoscaling data is usually collected under an existing policy, so it contains spurious correlations (for example, traffic and replica count rising together). This project learns a causal graph from the data with LiNGAM or NOTEARS, uses it to guide Q-learning agents, and compares them with standard agents trained on biased and unbiased datasets.

## Structure

```
agents/agents.py            Tabular and DQN Q-learning agents, plus their causal variants
causality/causal_discovery.py  Causal discovery wrapper (LiNGAM / NOTEARS)
envs/dataset_env.py         Offline environment built from a recorded service dataset
envs/autoscaler_env.py      Gymnasium environment with synthetic latency dynamics
sim_experiments/            Experiments on synthetic data (confounding and selection-bias scenarios)
run_evaluation.py           Main pipeline: causal discovery, training, and experiments
config.yaml                 Dataset paths, features, reward, and training settings
*.ipynb                     Dataset preprocessing, evaluation, and plotting
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install numpy pandas matplotlib networkx pyyaml torch gymnasium lingam causalnex
```

## Usage

1. Point `dataset.train_data_path` (biased data) and `dataset.eval_data_path` (unbiased data) in `config.yaml` to your CSV files. Datasets are not included in this repository.
2. Run the pipeline:

   ```bash
   python run_evaluation.py --config config.yaml
   ```

Each run writes to `results/<timestamp>/`:
- the discovered causal graph
- training curves
- the baseline comparison
- the results of the data-mixing (RD) and SLA threshold experiments
- a copy of the config that was used
