"""
dijkstra.py

File 2: Dijkstra Initialization cho mô hình Hybrid Routing IoT
- Load graph IoT đã xây dựng (node-link-weight)
- Tìm đường đi tối ưu QoS bằng Dijkstra
- Tính metric QoS của path
- Vẽ và lưu hình đường đi
- Xuất path cho GA / PSO
"""
import matplotlib
matplotlib.use("Agg")   # backend chỉ để lưu ảnh
import matplotlib.pyplot as plt

import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


# =========================
# CONFIG
# =========================
GRAPH_PATH = "./out/iot_graph.gpickle"
SAVE_FIG_PATH = "./out/dijkstra_path.png"

SOURCE_NODE = 0
TARGET_NODE = 24


# =========================
# LOAD GRAPH
# =========================
def load_graph(path: str) -> nx.Graph:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")

    with open(path, "rb") as f:
        G = pickle.load(f)

    return G


# =========================
# DIJKSTRA ROUTING
# =========================
def dijkstra_path(
    G: nx.Graph,
    source: int,
    target: int,
    weight: str = "cost"
) -> List[int]:
    return nx.shortest_path(G, source, target, weight=weight)


# =========================
# PATH METRICS
# =========================
def compute_path_metrics(
    G: nx.Graph,
    path: List[int]
) -> Dict[str, float]:
    total_cost = 0.0
    total_delay = 0.0
    total_energy = 0.0
    success_prob = 1.0  # PDR = product of link PDRs

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = G[u][v]

        total_cost += edge["cost"]
        total_delay += edge["delay"]
        total_energy += edge["energy"]
        success_prob *= edge["pdr"]

    return {
        "cost": total_cost,
        "delay": total_delay,
        "energy": total_energy,
        "pdr": success_prob
    }


# =========================
# VISUALIZATION
# =========================
def visualize_dijkstra_path(
    G: nx.Graph,
    path: List[int],
    save_path: str = None
):
    pos = nx.get_node_attributes(G, "pos")

    plt.figure(figsize=(8, 8))

    # draw base graph
    nx.draw_networkx_nodes(G, pos, node_size=120, node_color="lightgray")
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # highlight path
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="red", node_size=150)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=path_edges,
        edge_color="red",
        width=3
    )

    plt.title("Dijkstra Initialization Path (QoS-aware)")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("Loading IoT graph...")
    G = load_graph(GRAPH_PATH)

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    print(f"Running Dijkstra from {SOURCE_NODE} → {TARGET_NODE} ...")
    path = dijkstra_path(G, SOURCE_NODE, TARGET_NODE)

    metrics = compute_path_metrics(G, path)

    print("\n=== Dijkstra Result ===")
    print(f"Path: {path}")
    print(f"Total Cost   : {metrics['cost']:.4f}")
    print(f"Total Delay  : {metrics['delay']:.2f} ms")
    print(f"Total Energy : {metrics['energy']:.2f} mJ")
    print(f"End-to-end PDR: {metrics['pdr']:.4f}")

    print("\nVisualizing path...")
    visualize_dijkstra_path(G, path, SAVE_FIG_PATH)

    print("\nDone.")
