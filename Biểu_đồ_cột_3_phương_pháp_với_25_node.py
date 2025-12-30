"""
visualize.py
Visualization and comparison of QoS performance for IoT routing algorithms

Compared methods:
- Dijkstra (baseline)
- GA (multi-objective metaheuristic)
- Hybrid Dijkstra + GA + PSO (proposed)

Figures generated (for report / thesis):
1) Bar chart: End-to-end Delay comparison
2) Bar chart: Energy Consumption comparison
3) Bar chart: Packet Delivery Ratio (PDR) comparison
4) Bar chart: Throughput comparison
5) Network graph visualization: optimal paths of each method

Note:
- This is algorithm-level visualization, not packet-level animation.
- Suitable for "Mô phỏng và so sánh kết quả" section of the report.
"""

import pickle
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # non-GUI backend, tránh lỗi Tk/PIL
import matplotlib.pyplot as plt

# ================= CONFIG =================
GRAPH_PATH = "./out/iot_graph.gpickle"
RESULT_PATH = "./out/simulation_results.pkl"
GA_PARETO_PATH = "./out/ga_pareto_paths.pkl"
HYBRID_PARETO_PATH = "./out/hybrid_pareto_paths.pkl"  # optional
SRC = 0
DST = 24
# ==========================================


# ---------- LOAD DATA ----------
def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def load_results():
    with open(RESULT_PATH, "rb") as f:
        return pickle.load(f)


# ---------- BAR CHART UTILITY ----------
def plot_bar(title, ylabel, data_dict):
    methods = list(data_dict.keys())
    values = list(data_dict.values())

    plt.figure()
    plt.bar(methods, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"./out/{title.replace(' ', '_').lower()}.png", dpi=300)
    plt.close()


# ---------- PATH SELECTION FOR DRAWING ----------
def select_path(G, method):
    if method == "Dijkstra":
        return nx.shortest_path(G, SRC, DST, weight="delay")

    if method == "GA":
        with open(GA_PARETO_PATH, "rb") as f:
            pareto = pickle.load(f)
        # choose minimum energy path
        best = None
        best_energy = 1e9
        for p in pareto:
            energy = sum(G[p[i]][p[i+1]]["energy"] for i in range(len(p)-1))
            if energy < best_energy:
                best = p
                best_energy = energy
        return best

    if method == "Hybrid":
        try:
            with open(HYBRID_PARETO_PATH, "rb") as f:
                pareto = pickle.load(f)
        except FileNotFoundError:
            with open(GA_PARETO_PATH, "rb") as f:
                pareto = pickle.load(f)
        # choose minimum delay path
        best = None
        best_delay = 1e9
        for p in pareto:
            delay = sum(G[p[i]][p[i+1]]["delay"] for i in range(len(p)-1))
            if delay < best_delay:
                best = p
                best_delay = delay
        return best


# ---------- NETWORK VISUALIZATION ----------
def draw_network(G, paths):
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, node_size=300, alpha=0.6, with_labels=True)

    for method, path in paths.items():
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=3, label=method)

    plt.title("Optimal Routing Paths Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./out/network_paths_comparison.png", dpi=300)
    plt.close()
    plt.savefig(f"./out/{title.replace(' ', '_').lower()}.png", dpi=300)
    plt.close()


# ================= MAIN =================
if __name__ == "__main__":
    G = load_graph()
    results = load_results()

    # ---------- BAR CHARTS ----------
    plot_bar(
        "End-to-End Delay Comparison",
        "Delay (ms)",
        {m: results[m]["Delay"] for m in results}
    )

    plot_bar(
        "Energy Consumption Comparison",
        "Energy (mJ)",
        {m: results[m]["Energy"] for m in results}
    )

    plot_bar(
        "Packet Delivery Ratio (PDR) Comparison",
        "PDR",
        {m: results[m]["PDR"] for m in results}
    )

    plot_bar(
        "Throughput Comparison",
        "Packets Delivered",
        {m: results[m]["Throughput"] for m in results}
    )

    # ---------- NETWORK GRAPH ----------
    paths = {
        "Dijkstra": select_path(G, "Dijkstra"),
        "GA": select_path(G, "GA"),
        "Hybrid": select_path(G, "Hybrid"),
    }

    draw_network(G, paths)

    print("Visualization completed. Figures saved to ./out/")
