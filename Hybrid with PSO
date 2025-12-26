"""
PSO_multiobjective.py (GA–PSO Hybrid)
Multi-objective PSO for QoS-aware IoT Routing
Role: Refinement & Stabilization of GA Pareto Paths
Objectives: Delay ↓, Energy ↓, (1-PDR) ↓
"""

import random
import networkx as nx
import pickle
from typing import List, Tuple

# ================= CONFIG =================
GRAPH_PATH = "./out/iot_graph.gpickle"
GA_PARETO_PATH = "./out/ga_pareto_paths.pkl"  # output from GA
SRC = 0
DST = 24

N_PARTICLES = 20
N_ITER = 25

# Adaptive PSO parameters
W_MAX, W_MIN = 0.9, 0.3
C1_MAX, C1_MIN = 2.0, 0.5
C2_MAX, C2_MIN = 2.0, 0.5

MAX_ARCHIVE = 6   # small & clean Pareto set

ALPHA = 0.4
BETA = 0.3
GAMMA = 0.3
# ==========================================


# ---------- LOAD GRAPH ----------
def load_graph(path: str) -> nx.Graph:
    print("Loading IoT graph...")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------- METRICS ----------
def path_metrics(G: nx.Graph, path: List[int]) -> Tuple[float, float, float]:
    delay = energy = 0.0
    pdr = 1.0
    for i in range(len(path) - 1):
        if not G.has_edge(path[i], path[i + 1]):
            return 1e6, 1e6, 0.0
        e = G[path[i]][path[i + 1]]
        delay += e["delay"]
        energy += e["energy"]
        pdr *= e["pdr"]
    return delay, energy, pdr


def fitness_vector(G, path):
    if path[0] != SRC or path[-1] != DST:
        return (1e6, 1e6, 1.0)
    d, e, p = path_metrics(G, path)
    return (d, e, 1 - p)


def composite_cost(delay, energy, pdr):
    return ALPHA * delay + BETA * energy + GAMMA * (1 - pdr)


# ---------- DOMINANCE ----------
def dominates(a, b) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


# ---------- PATH OPERATORS ----------
def refine_path(G, path):
    if len(path) < 4:
        return path
    i = random.randint(1, len(path) - 3)
    neighbors = list(G.neighbors(path[i]))
    random.shuffle(neighbors)
    for n in neighbors:
        if n in path:
            continue
        try:
            tail = nx.shortest_path(G, n, DST, weight="delay")
            new_path = path[:i] + [n] + tail[1:]
            if len(new_path) == len(set(new_path)):
                return new_path
        except:
            pass
    return path


# ---------- LOAD GA PARETO ----------
def load_ga_pareto():
    with open(GA_PARETO_PATH, "rb") as f:
        return pickle.load(f)


# ================= MAIN =================
if __name__ == "__main__":
    G = load_graph(GRAPH_PATH)
    print(f"Running Hybrid GA–PSO from {SRC} → {DST}")

    # --- Initialize swarm from GA Pareto ---
    ga_pareto = load_ga_pareto()
    swarm = ga_pareto.copy()
    while len(swarm) < N_PARTICLES:
        swarm.append(refine_path(G, random.choice(ga_pareto)))

    pbest = swarm.copy()
    pbest_fit = [fitness_vector(G, p) for p in pbest]

    archive = []

    for it in range(N_ITER):
        w = W_MAX - (W_MAX - W_MIN) * it / N_ITER
        c1 = C1_MAX - (C1_MAX - C1_MIN) * it / N_ITER
        c2 = C2_MIN + (C2_MAX - C2_MIN) * it / N_ITER

        fits = [fitness_vector(G, p) for p in swarm]

        # Update pBest
        for i in range(len(swarm)):
            if dominates(fits[i], pbest_fit[i]):
                pbest[i] = swarm[i]
                pbest_fit[i] = fits[i]

        # Update Pareto archive (unique only)
        for p, f in zip(swarm, fits):
            if any(p == pa for pa, _ in archive):
                continue
            if not any(dominates(f2, f) for _, f2 in archive):
                archive = [(pa, fa) for pa, fa in archive if not dominates(f, fa)]
                archive.append((p, f))

        if len(archive) > MAX_ARCHIVE:
            archive = archive[:MAX_ARCHIVE]

        # Select gBest
        gbest, _ = random.choice(archive)

        # Refine swarm
        new_swarm = []
        for i in range(len(swarm)):
            p = swarm[i]
            if random.random() < c1:
                p = refine_path(G, pbest[i])
            if random.random() < c2:
                p = refine_path(G, gbest)
            new_swarm.append(p)
        swarm = new_swarm

        print(f"Iter {it:02d} | Archive size: {len(archive)}")


# ---------- FINAL OUTPUT ----------
print("\n===== FINAL PARETO FRONT (HYBRID GA–PSO) =====")

for idx, (p, _) in enumerate(archive):
    d, e, pdr = path_metrics(G, p)
    cost = composite_cost(d, e, pdr)
    print(f"\nSolution {idx+1}")
    print("Path            :", p)
    print(f"Total Delay     : {d:.2f} ms")
    print(f"Total Energy    : {e:.2f} mJ")
    print(f"End-to-end PDR  : {pdr:.4f}")
    print(f"Composite Cost  : {cost:.4f}")
