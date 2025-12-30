"""
GA_multiobjective.py
Multi-objective Genetic Algorithm for QoS-aware IoT Routing
Objectives: Delay ↓, Energy ↓, (1-PDR) ↓
"""

import random
import networkx as nx
from typing import List, Tuple
import pickle

# ================= CONFIG =================
GRAPH_PATH = "./out/iot_graph.gpickle"
SRC = 0
DST = 24

POP_SIZE = 30
N_GEN = 30
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.5
ELITISM = True

ALPHA = 0.4
BETA = 0.3
GAMMA = 0.3
# ==========================================


# ---------- LOAD GRAPH ----------
def load_graph(path: str) -> nx.Graph:
    print("Loading IoT graph...")
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G


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


def composite_cost(delay, energy, pdr):
    return ALPHA * delay + BETA * energy + GAMMA * (1 - pdr)


# ---------- DOMINANCE ----------
def dominates(a, b) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


# ---------- FITNESS ----------
def fitness_vector(G, path):
    if path[0] != SRC or path[-1] != DST:
        return (1e6, 1e6, 1.0)

    delay, energy, pdr = path_metrics(G, path)
    return (delay, energy, 1 - pdr)


# ---------- GA OPERATORS ----------
def tournament_selection(pop, fits):
    cand = random.sample(list(zip(pop, fits)), TOURNAMENT_K)
    best = cand[0]
    for c in cand[1:]:
        if dominates(c[1], best[1]):
            best = c
    return best[0]


def crossover(p1, p2):
    common = set(p1[1:-1]) & set(p2[1:-1])
    if not common:
        return p1.copy()

    cut = random.choice(list(common))
    i, j = p1.index(cut), p2.index(cut)
    child = p1[:i] + p2[j:]

    return child if len(child) == len(set(child)) else p1.copy()


def mutate(G, path):
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


# ---------- INITIAL POP ----------
def init_population(G):
    pop = []

    try:
        paths = nx.shortest_simple_paths(G, SRC, DST, weight="delay")
        for p in paths:
            pop.append(list(p))
            if len(pop) >= POP_SIZE:
                break
    except:
        pass

    if not pop:
        pop.append(nx.shortest_path(G, SRC, DST, weight="delay"))

    print(f"Initial population size: {len(pop)}")
    return pop


# ================= MAIN =================
if __name__ == "__main__":
    G = load_graph(GRAPH_PATH)
    print(f"Running Multi-objective GA from {SRC} → {DST}")

    population = init_population(G)

    for gen in range(N_GEN):
        fits = [fitness_vector(G, p) for p in population]

        pareto = []
        for i, f in enumerate(fits):
            if not any(dominates(f2, f) for j, f2 in enumerate(fits) if i != j):
                pareto.append(population[i])

        print(f"Gen {gen:02d} | Pareto solutions: {len(pareto)}")

        new_pop = pareto.copy() if ELITISM else []

        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(population, fits)
            p2 = tournament_selection(population, fits)

            child = p1.copy()
            if random.random() < CROSSOVER_RATE:
                child = crossover(p1, p2)

            if random.random() < MUTATION_RATE:
                child = mutate(G, child)

            if child not in new_pop:
                new_pop.append(child)

        population = new_pop[:POP_SIZE]


# ---------- FINAL OUTPUT ----------
print("\n===== FINAL PARETO FRONT (MULTI-OBJECTIVE GA) =====")

final_fits = [(p, fitness_vector(G, p)) for p in population]
final_pareto = []

for p, f in final_fits:
    if not any(dominates(f2, f) for _, f2 in final_fits if f2 != f):
        final_pareto.append(p)

final_pareto = final_pareto[:5]   # chỉ in nghiệm mạnh

for idx, p in enumerate(final_pareto):
    delay, energy, pdr = path_metrics(G, p)
    cost = composite_cost(delay, energy, pdr)

    print(f"\nSolution {idx+1}")
    print("Path            :", p)
    print(f"Total Delay     : {delay:.2f} ms")
    print(f"Total Energy    : {energy:.2f} mJ")
    print(f"End-to-end PDR  : {pdr:.4f}")
    print(f"Composite Cost  : {cost:.4f}")

GA_PARETO_PATH = "./out/ga_pareto_paths.pkl"

with open(GA_PARETO_PATH, "wb") as f:
    pickle.dump(final_pareto, f)

print(f"\nGA Pareto paths saved to {GA_PARETO_PATH}")
