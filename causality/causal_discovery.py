import pandas as pd
import numpy as np
import lingam
from lingam.utils import make_dot
import matplotlib.pyplot as plt
import networkx as nx
import os

try:
    from causalnex.structure.notears import from_pandas as causalnex_from_pandas
except ImportError:
    causalnex_from_pandas = None

class CausalDiscovery:
    def __init__(self, method='lingam'):
        self.method = method
        self.model = None
        self.adjacency_matrix_ = None
        self.labels_ = None

    def fit(
        self,
        df,
        prior_knowledge=None,
        notears_w_threshold=0.01,
        notears_max_iter=100,
        tabu_edges=None,
        fixed_edges=None,
    ):
        """
        Fit causal model to dataframe.

        Supported methods:
        - lingam: DirectLiNGAM, optional `prior_knowledge`
        - notears: CausalNex NOTEARS (from_pandas). Can provide `fixed_edges` (list of valid (source, target) tuples) to calculate weights for a known structure.

        Assumption: Data is continuous.
        """
        self.labels_ = df.columns.tolist()
        
        if self.method == 'lingam':
            self.model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
            self.model.fit(df)
            self.adjacency_matrix_ = self.model.adjacency_matrix_
        elif self.method == 'notears':
            if causalnex_from_pandas is None:
                raise ImportError(
                    "causalnex is not installed. Install causalnex to use method='notears'."
                )
                
            _tabu_edges = list(tabu_edges) if tabu_edges else []
            if fixed_edges is not None:
                import itertools
                all_edges = set((u, v) for u, v in itertools.product(self.labels_, self.labels_) if u != v)
                forbidden = all_edges - set(tuple(e) for e in fixed_edges)
                _tabu_edges.extend(list(forbidden))

            self.model = causalnex_from_pandas(
                df,
                tabu_edges=_tabu_edges if _tabu_edges else None,
                max_iter=int(notears_max_iter),
                w_threshold=float(notears_w_threshold),
            )

            # Keep the same orientation used by LiNGAM: adj[target, source].
            n = len(self.labels_)
            idx = {label: i for i, label in enumerate(self.labels_)}
            adj = np.zeros((n, n), dtype=float)
            for source, target, data in self.model.edges(data=True):
                s = idx[source]
                t = idx[target]
                adj[t, s] = float(data.get('weight', 0.0))
            self.adjacency_matrix_ = adj
        else:
            raise NotImplementedError(f"Method {self.method} not implemented.")
            
        return self

    def visualize(self, df=None, save_path='causal_graph.png', standardize=True):
        """
        Visualize the learned graph.
        If standardize=True and df provided, shows Beta coefficients.
        """
        if self.adjacency_matrix_ is None:
            raise ValueError("Model not fitted yet.")
            
        adj = self.adjacency_matrix_.copy()
        
        # Standardize weights if requested
        if standardize and df is not None:
            print("Visualizing Standardized Coefficients (Beta)...")
            stds = df.std()
            for i, target in enumerate(self.labels_):
                for j, source in enumerate(self.labels_):
                    if abs(adj[i, j]) > 0.01:
                        # Beta = b * (std_x / std_y)
                        adj[i, j] = adj[i, j] * (stds[source] / stds[target])
        
        # method 1: networkx
        G = nx.DiGraph()
        n_nodes = len(self.labels_)
        for i in range(n_nodes):
            G.add_node(self.labels_[i])
            for j in range(n_nodes):
                weight = adj[i, j]
                if abs(weight) > 0.01: # Threshold
                    G.add_edge(self.labels_[j], self.labels_[i], weight=weight)
        
        plt.figure(figsize=(6, 4))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=16, arrowsize=20)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        # Round weights for display
        edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        title = f"Causal Graph ({self.method})"
        if standardize:
            title += " [Standardized Coefficients]"
        plt.title(title)
        
        plt.savefig(save_path)
        print(f"Saved graph visualization to {save_path}")
        plt.close()

def run_discovery_scenario(scenario_file):
    print(f"--- Discovering Causal Graph for {scenario_file} ---")
    if not os.path.exists(scenario_file):
        print(f"File {scenario_file} not found. Please run data_loader.py first.")
        return

    df = pd.read_csv(scenario_file)
    cd = CausalDiscovery()
    cd.fit(df)
    
    # Visualise with Standardization
    vis_name = scenario_file.replace('.csv', '_graph.png')
    cd.visualize(df=df, save_path=vis_name, standardize=True)

if __name__ == "__main__":
    if os.path.exists('data/dataset_scenario_1.csv'):
        run_discovery_scenario('data/dataset_scenario_1.csv')
    if os.path.exists('data/dataset_scenario_2.csv'):
        run_discovery_scenario('data/dataset_scenario_2.csv')
