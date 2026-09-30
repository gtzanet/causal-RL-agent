import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from causalnex.structure.notears import from_pandas

from causality.causal_discovery import CausalDiscovery


def build_columns(service_id: int):
    prefix = str(service_id)
    return [f"{prefix}_arr", f"{prefix}_rep", f"{prefix}_lat"]


def build_expected_prior_knowledge():
        """
        For columns [arr, rep, lat], enforce:
            arr -> lat
            rep -> lat
        and forbid direct links between arr and rep and reverse links from lat.

        LiNGAM convention used here (matching adjacency interpretation): pk[to, from]
            1  : directed path exists
            0  : directed path does NOT exist
            -1 : unknown
        """
        pk = np.full((3, 3), -1, dtype=int)

        arr_i, rep_i, lat_i = 0, 1, 2

        # Force expected edges
        pk[lat_i, arr_i] = 1
        pk[lat_i, rep_i] = 1

        # Forbid reverse from latency
        pk[arr_i, lat_i] = 0
        pk[rep_i, lat_i] = 0

        # Forbid direct arr <-> rep
        pk[arr_i, rep_i] = 0
        pk[rep_i, arr_i] = 0

        return pk


def main():
    parser = argparse.ArgumentParser(
        description="Build and plot a causal graph for one service using arr/rep/lat variables."
    )
    parser.add_argument(
        "--service",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Service id in shortnames dataset (1=frame-sampler, 2=frame-resizer, 3=object-detector)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/real_dataset_260313_shortnames.csv",
        help="Input shortnames CSV",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Output subset CSV path (default: data/real_dataset_260313_<service>_only.csv)",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default=None,
        help="Output graph PNG path (default: plots/real_dataset_260313_<service>_only_graph_usercode.png)",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.01,
        help="Threshold for printing edges from adjacency matrix",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="lingam",
        choices=["lingam", "notears"],
        help="Causal discovery method.",
    )
    parser.add_argument(
        "--enforce-expected",
        action="store_true",
        help="Enforce expected structure: <service>_arr -> <service>_lat and <service>_rep -> <service>_lat.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input)
    service = args.service
    columns = build_columns(service)

    out_csv = Path(args.out_csv) if args.out_csv else Path(f"data/real_dataset_260313_{service}_only.csv")
    out_png = (
        Path(args.out_png)
        if args.out_png
        else Path(f"plots/real_dataset_260313_{service}_only_graph_usercode.png")
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {input_csv}: {missing}. Expected shortnames format with columns like '2_arr', '2_rep', '2_lat'."
        )

    df_service = df[columns].copy()
    df_service.to_csv(out_csv, index=False)

    if args.method == "lingam":
        cd = CausalDiscovery(method="lingam")
        prior_knowledge = build_expected_prior_knowledge() if args.enforce_expected else None
        cd.fit(df_service, prior_knowledge=prior_knowledge)
        cd.visualize(df=df_service, save_path=str(out_png), standardize=True)
        labels = cd.labels_
        adj = cd.adjacency_matrix_
    else:
        tabu_edges = None
        if args.enforce_expected:
            arr, rep, lat = columns
            tabu_edges = [
                (lat, arr),
                (lat, rep),
                (arr, rep),
                (rep, arr),
            ]
        sm = from_pandas(df_service, w_threshold=0.0, tabu_edges=tabu_edges)

        labels = columns
        adj = np.zeros((len(labels), len(labels)), dtype=float)
        for src, tgt, data in sm.edges(data=True):
            j = labels.index(src)
            i = labels.index(tgt)
            adj[i, j] = float(data.get("weight", 0.0))

        # Plot NOTEARS graph
        G = nx.DiGraph()
        for lbl in labels:
            G.add_node(lbl)
        for i, tgt in enumerate(labels):
            for j, src in enumerate(labels):
                w = adj[i, j]
                if abs(w) > args.edge_threshold:
                    G.add_edge(src, tgt, weight=w)

        plt.figure(figsize=(6, 4))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=900,
            font_size=10,
            arrowsize=18,
        )
        edge_labels = nx.get_edge_attributes(G, "weight")
        edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        plt.title("Causal Graph (notears)")
        plt.savefig(out_png)
        plt.close()

    print(f"Input: {input_csv}")
    print(f"Subset CSV: {out_csv}")
    print(f"Graph PNG: {out_png}")
    print(f"Method: {args.method}")
    print("Columns:", columns)
    if args.enforce_expected:
        print("Using prior knowledge: enforce arr->lat and rep->lat")
    print("Adjacency matrix:")
    print(adj)

    print(f"Edges |w| > {args.edge_threshold}:")
    found = 0
    for i, target in enumerate(labels):
        for j, source in enumerate(labels):
            w = adj[i, j]
            if abs(w) > args.edge_threshold:
                found += 1
                print(f"{source} -> {target}: {w:.6f}")
    if found == 0:
        print("(none)")


if __name__ == "__main__":
    main()
