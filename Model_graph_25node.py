"""
iot_graph_full.py

Phiên bản FULL: Xây dựng mô hình mạng IoT dạng đồ thị (node, link, weight) chuẩn cho đồ án
"Hybrid QoS-aware Routing in IoT using Dijkstra Initialization and GA/PSO Optimization".

Chức năng:
- Cấu hình mô phỏng (seed, số node, khu vực, tx range, trọng số QoS)
- Tạo vị trí node (reproducible bằng seed)
- Sinh cạnh theo bán kính truyền thông
- Tính các tham số QoS cho mỗi cạnh: delay (ms), energy (mJ), bandwidth (kbps), PER, PDR
- Chuẩn hóa và tính cost tổng hợp (fitness cost) theo hệ số trọng số
- Visualize (matplotlib) đồ thị, hiển thị trọng số
- Lưu / tải graph (networkx gpickle)
- Lấy đường đi Dijkstra theo cost tổng hợp
- Sinh quần thể khởi tạo cho GA/PSO từ Dijkstra (perturbations)

Yêu cầu: networkx, numpy, matplotlib
pip install networkx numpy matplotlib

"""
import matplotlib
matplotlib.use("Agg")
from typing import Tuple, Dict, Any, List
import random
import math
import json
import os
import pickle

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Cấu hình mặc định 
# ----------------------------
DEFAULT_CONFIG = {
    "num_nodes": 30,
    "area_size": 100.0,        # đơn vị mét
    "tx_range": 30.0,         # bán kính truyền (m)
    "seed": 42,
    # QoS weight coefficients (dùng để tính cost tổng hợp)
    "w_delay": 0.4,
    "w_energy": 0.3,
    "w_pdr": 0.3,
    # random parameters for synthetic QoS
    "delay_random_min": 1.0,
    "delay_random_max": 5.0,
    "energy_coef_min": 0.03,   # energy per meter coefficient
    "energy_coef_max": 0.12,
    "bandwidth_min": 50.0,     # kbps
    "bandwidth_max": 250.0,
    "per_min": 0.005,
    "per_max": 0.15
}


class IoTGraphBuilder:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        # reproducible
        random.seed(self.config["seed"]) 
        np.random.seed(self.config["seed"])
        self.G = nx.Graph()

    def generate_positions(self) -> Dict[int, Tuple[float, float]]:
        N = self.config["num_nodes"]
        L = self.config["area_size"]
        positions = {i: (random.uniform(0, L), random.uniform(0, L)) for i in range(N)}
        return positions

    @staticmethod
    def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def build_graph(self) -> nx.Graph:
        """Sinh đồ thị với các thuộc tính cạnh (delay, energy, bandwidth, PER, PDR)"""
        pos = self.generate_positions()
        G = nx.Graph()
        for n, p in pos.items():
            G.add_node(n, pos=p)

        tx = self.config["tx_range"]
        for i in range(self.config["num_nodes"]):
            for j in range(i + 1, self.config["num_nodes"]):
                d = self.euclidean(pos[i], pos[j])
                if d <= tx:
                    # synthetic QoS attributes
                    # delay: base proportional to distance plus random jitter (ms)
                    delay = d / 5.0 + random.uniform(self.config["delay_random_min"], self.config["delay_random_max"]) 
                    # energy: simple model (mJ) proportional to distance
                    coef = random.uniform(self.config["energy_coef_min"], self.config["energy_coef_max"])
                    energy = d * coef
                    # bandwidth: higher when distance smaller
                    bandwidth = max(self.config["bandwidth_min"], self.config["bandwidth_max"] - d * 1.5)
                    # PER: packet error rate increases with distance
                    per = min(self.config["per_max"], self.config["per_min"] + (d / tx) * self.config["per_max"]) 
                    pdr = max(0.0, 1.0 - per)

                    G.add_edge(i, j, distance=d, delay=delay, energy=energy, bandwidth=bandwidth, per=per, pdr=pdr)

        self.G = G
        return G

    def normalize_edge_attributes(self, attr_names: List[str]) -> Dict[str, Dict[Tuple[int, int], float]]:
        """Chuẩn hoá các thuộc tính cạnh theo khoảng [0,1] cho từng attribute tên trong attr_names.
        Trả về dict mapping attr -> {(u,v): normalized_value}
        """
        result = {}
        for attr in attr_names:
            vals = [d.get(attr, 0.0) for _, _, d in self.G.edges(data=True)]
            if len(vals) == 0:
                result[attr] = {}
                continue
            vmin, vmax = min(vals), max(vals)
            # avoid divide by zero
            denom = vmax - vmin if vmax != vmin else 1.0
            normalized = {}
            for u, v, d in self.G.edges(data=True):
                raw = d.get(attr, 0.0)
                normalized[(u, v)] = (raw - vmin) / denom
            result[attr] = normalized
        return result

    def compute_composite_cost(self) -> None:
        """Tính cost tổng hợp gán vào thuộc tính 'cost' cho mỗi cạnh.
        Cost = w_delay*norm(delay) + w_energy*norm(energy) + w_pdr*(1 - norm(pdr))
        Lưu ý: pdr càng lớn -> cost càng thấp, nên dùng 1 - norm(pdr)
        """
        weights = self.normalize_edge_attributes(["delay", "energy", "pdr"])
        w_delay = self.config["w_delay"]
        w_energy = self.config["w_energy"]
        w_pdr = self.config["w_pdr"]

        for u, v, d in self.G.edges(data=True):
            ndelay = weights["delay"].get((u, v), 0.0)
            nenergy = weights["energy"].get((u, v), 0.0)
            npdr = weights["pdr"].get((u, v), 0.0)
            cost = w_delay * ndelay + w_energy * nenergy + w_pdr * (1.0 - npdr)
            # store also normalized fields for debugging/plotting
            d["ndelay"] = ndelay
            d["nenergy"] = nenergy
            d["npdr"] = npdr
            d["cost"] = cost

    # ------------------ Utility / I/O ------------------
    def save_graph(self, path: str) -> None:
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path, "wb") as f:
            pickle.dump(self.G, f)

    def load_graph(self, path: str) -> nx.Graph:
        with open(path, "rb") as f:
            self.G = pickle.load(f)
        return self.G

    def visualize(self, show_edge_labels: bool = False, save_path: str = None) -> None:
        """Vẽ đồ thị IoT với vị trí node và màu cạnh theo cost."""
        if self.G is None or len(self.G) == 0:
            raise RuntimeError("Graph is empty. Build graph first.")

        pos = nx.get_node_attributes(self.G, "pos")
        plt.figure(figsize=(8, 8))
        nodes = nx.draw_networkx_nodes(self.G, pos, node_size=120, node_color="skyblue")
        nx.draw_networkx_labels(self.G, pos, font_size=8)

        # edge colors by cost
        costs = [d.get("cost", 0.0) for _, _, d in self.G.edges(data=True)]
        # normalize for colormap
        if len(costs) > 0:
            vmin, vmax = min(costs), max(costs)
            denom = vmax - vmin if vmax != vmin else 1.0
            colors = [(c - vmin) / denom for c in costs]
        else:
            colors = [0 for _ in costs]

        edges = nx.draw_networkx_edges(self.G, pos, edge_color=colors, edge_cmap=plt.cm.viridis, width=2)
        plt.colorbar(edges)

        if show_edge_labels:
            edge_labels = {(u, v): f"{d['cost']:.2f}" for u, v, d in self.G.edges(data=True)}
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=6)

        plt.axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    # ------------------ Dijkstra & path helpers ------------------
    def dijkstra_path(self, source: int, target: int, weight_attr: str = "cost") -> List[int]:
        if source not in self.G or target not in self.G:
            raise ValueError("Source or target not in graph")
        # networkx shortest_path handles 'weight' attribute name
        path = nx.shortest_path(self.G, source=source, target=target, weight=weight_attr)
        return path

    def path_cost(self, path: List[int], attr: str = "cost") -> float:
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total += self.G[u][v].get(attr, 0.0)
        return total

    def all_pairs_shortest_paths_cost(self, attr: str = "cost") -> Dict[Tuple[int, int], float]:
        # compute shortest path lengths for all-pairs using given weight
        length = dict(nx.all_pairs_dijkstra_path_length(self.G, weight=attr))
        result = {}
        for u in length:
            for v, val in length[u].items():
                result[(u, v)] = val
        return result

    # ------------------ Initial population generation ------------------
    def perturb_path(self, path: List[int], max_attempts: int = 10) -> List[int]:
        """Tạo biến thể của path bằng cách thay một đọan con bằng đường ngắn nhất cục bộ.
        Phương pháp:
        - chọn 2 điểm i<j trong path, tìm shortest path giữa path[i] và path[j] bằng weight 'cost',
          thay thế đoạn cũ bằng đoạn mới (nếu hợp lệ và không lặp).
        """
        if len(path) < 4:
            return path.copy()
        for _ in range(max_attempts):
            i = random.randint(0, len(path) - 3)
            j = random.randint(i + 2, len(path) - 1)
            a, b = path[i], path[j]
            try:
                sub = nx.shortest_path(self.G, source=a, target=b, weight="cost")
            except nx.NetworkXNoPath:
                continue
            # build new path
            new_path = path[:i] + sub + path[j + 1:]
            # check loop-free and valid
            if len(new_path) == len(set(new_path)):
                return new_path
        return path.copy()

    def generate_initial_population(self, source: int, target: int, pop_size: int = 20) -> List[List[int]]:
        """Sinh quần thể cho GA/PSO:
        - 1 cá thể đầu là đường Dijkstra
        - phần còn lại là biến thể perturb (hoán đổi, chèn, subpath replacement)
        """
        pop = []
        try:
            base = self.dijkstra_path(source, target, weight_attr="cost")
        except Exception:
            # fallback: shortest by distance
            base = nx.shortest_path(self.G, source=source, target=target, weight="distance")

        pop.append(base)
        attempts = 0
        while len(pop) < pop_size and attempts < pop_size * 10:
            candidate = self.perturb_path(base)
            if candidate not in pop:
                pop.append(candidate)
            attempts += 1

        # if still small, fill with random simple paths using DFS up to length limit
        if len(pop) < pop_size:
            nodes = list(self.G.nodes())
            while len(pop) < pop_size:
                try:
                    rpath = nx.shortest_path(self.G, source=source, target=target)
                    if rpath not in pop:
                        pop.append(rpath)
                except nx.NetworkXNoPath:
                    break
        return pop


# ----------------------------
# CLI / Example usage
# ----------------------------
if __name__ == "__main__":
    cfg = {
        "num_nodes": 25,
        "area_size": 120.0,
        "tx_range": 30.0,
        "seed": 123,
        "w_delay": 0.4,
        "w_energy": 0.35,
        "w_pdr": 0.25
    }

    builder = IoTGraphBuilder(cfg)
    G = builder.build_graph()
    builder.compute_composite_cost()

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # save graph
    builder.save_graph("./out/iot_graph.gpickle")

    # visualize with edge cost labels
    builder.visualize(show_edge_labels=True, save_path="./out/iot_graph.png")

    # example dijkstra
    nodes = list(G.nodes())
    if len(nodes) >= 2:
        s, t = nodes[0], nodes[-1]
        path = builder.dijkstra_path(s, t)
        cost = builder.path_cost(path)
        print(f"Dijkstra path {s} -> {t}: {path}, cost={cost:.4f}")

        pop = builder.generate_initial_population(s, t, pop_size=12)
        print("Initial population (paths):")
        for p in pop:
            print(p)

    print("Done.")
