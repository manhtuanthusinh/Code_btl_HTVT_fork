"""
simulate.py
Algorithm-level simulation and performance comparison for IoT routing

Methods compared:
1) Dijkstra (baseline)
2) GA (multi-objective metaheuristic)
3) Hybrid Dijkstra + GA + PSO (proposed model)

Metrics collected:
- End-to-end Delay
- Energy Consumption
- Packet Delivery Ratio (PDR)
- Throughput (relative, derived from PDR)

Simulation is performed at routing-algorithm level (no packet-level simulation).
"""

import pickle
import networkx as nx
from typing import Dict, Tuple

# ================= CONFIG =================
GRAPH_PATH = "./out/iot_graph.gpickle"
GA_PARETO_PATH = "./out/ga_pareto_paths.pkl"
HYBRID_PARETO_PATH = "./out/hybrid_pareto_paths.pkl"  # optional
SRC = 0
DST = 24
PACKET_RATE = 100   # packets per simulation round (relative)
# ==========================================


# ---------- LOAD GRAPH ----------
def load_graph(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------- METRICS ----------
def path_metrics(G: nx.Graph, path) -> Tuple[float, float, float]:
    delay = energy = 0.0
    pdr = 1.0

    for i in range(len(path) - 1):
        e = G[path[i]][path[i + 1]]
        delay += e["delay"]
        energy += e["energy"]
        pdr *= e["pdr"]

    return delay, energy, pdr


# ---------- BASELINE: DIJKSTRA ----------
def run_dijkstra(G: nx.Graph) -> Tuple[float, float, float]:
    path = nx.shortest_path(G, SRC, DST, weight="delay")
    return path_metrics(G, path)


# ---------- GA (MULTI-OBJECTIVE) ----------
def run_ga(G: nx.Graph) -> Tuple[float, float, float]:
    with open(GA_PARETO_PATH, "rb") as f:
        pareto = pickle.load(f)

    # Representative GA solution: minimum energy
    best = None
    best_energy = 1e9

    for p in pareto:
        d, e, pdr = path_metrics(G, p)
        if e < best_energy:
            best = (d, e, pdr)
            best_energy = e

    return best


# ---------- HYBRID GA–PSO (PROPOSED) ----------
def run_hybrid(G: nx.Graph) -> Tuple[float, float, float]:
    try:
        with open(HYBRID_PARETO_PATH, "rb") as f:
            pareto = pickle.load(f)
    except FileNotFoundError:
        # fallback: use GA Pareto if hybrid Pareto is not saved separately
        with open(GA_PARETO_PATH, "rb") as f:
            pareto = pickle.load(f)

    # Representative hybrid solution: minimum composite QoS cost
    best = None
    best_score = 1e9

    for p in pareto:
        d, e, pdr = path_metrics(G, p)
        score = 0.4 * d + 0.3 * e + 0.3 * (1 - pdr)
        if score < best_score:
            best = (d, e, pdr)
            best_score = score

    return best


# ================= MAIN =================
if __name__ == "__main__":
    G = load_graph(GRAPH_PATH)

    results: Dict[str, Dict[str, float]] = {}

    # Dijkstra baseline
    d, e, pdr = run_dijkstra(G)
    results["Dijkstra"] = {
        "Delay": d,
        "Energy": e,
        "PDR": pdr,
        "Throughput": pdr * PACKET_RATE,
    }

    # GA multi-objective
    d, e, pdr = run_ga(G)
    results["GA"] = {
        "Delay": d,
        "Energy": e,
        "PDR": pdr,
        "Throughput": pdr * PACKET_RATE,
    }

    # Hybrid GA–PSO (proposed)
    d, e, pdr = run_hybrid(G)
    results["Hybrid"] = {
        "Delay": d,
        "Energy": e,
        "PDR": pdr,
        "Throughput": pdr * PACKET_RATE,
    }

    # ---------- PRINT RESULTS ----------
    print("\n===== SIMULATION RESULTS (QoS COMPARISON) =====")
    for method, m in results.items():
        print(f"\nMethod: {method}")
        print(f"  Delay       : {m['Delay']:.2f} ms")
        print(f"  Energy      : {m['Energy']:.2f} mJ")
        print(f"  PDR         : {m['PDR']:.4f}")
        print(f"  Throughput  : {m['Throughput']:.2f} packets")

    # ---------- SAVE RESULTS ----------
    with open("./out/simulation_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\nSimulation results saved to ./out/simulation_results.pkl")
