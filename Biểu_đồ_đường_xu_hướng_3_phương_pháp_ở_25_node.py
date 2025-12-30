"""
visualize_trend.py
Generate combined line-trend plots for QoS comparison

Purpose:
- Bổ sung cho visualize.py (KHÔNG thay thế, KHÔNG ghi đè)
- Sinh đồ thị dạng đường (line chart) so sánh xu hướng giữa:
  Dijkstra – GA – Hybrid (Dijkstra+GA+PSO)

Input:
- ./out/simulation_results.pkl (đã sinh từ simulate.py)

Output (NEW files):
- ./out/trend_delay_comparison.png
- ./out/trend_energy_comparison.png
- ./out/trend_pdr_comparison.png
- ./out/trend_overall_qos.png
"""

import pickle
import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt

# ================= CONFIG =================
RESULT_PATH = "./out/simulation_results.pkl"
OUT_DIR = "./out/"
# ==========================================


# ---------- LOAD RESULTS ----------
def load_results():
    with open(RESULT_PATH, "rb") as f:
        return pickle.load(f)


# ---------- LINE PLOT ----------
def plot_line(title, ylabel, methods, values, filename):
    plt.figure(figsize=(7, 5))
    plt.plot(methods, values, marker='o', linewidth=2)
    plt.title(title)
    plt.xlabel("Routing Method")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUT_DIR + filename, dpi=300)
    plt.close()


# ---------- OVERALL TREND ----------
def plot_overall_trend(methods, results):
    plt.figure(figsize=(8, 6))

    plt.plot(methods, [results[m]["Delay"] for m in methods], marker='o', label="Delay")
    plt.plot(methods, [results[m]["Energy"] for m in methods], marker='s', label="Energy")
    plt.plot(methods, [results[m]["PDR"] for m in methods], marker='^', label="PDR")

    plt.title("Overall QoS Trend Comparison")
    plt.xlabel("Routing Method")
    plt.ylabel("QoS Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUT_DIR + "trend_overall_qos.png", dpi=300)
    plt.close()


# ================= MAIN =================
if __name__ == "__main__":
    results = load_results()

    methods = list(results.keys())

    # Delay trend
    plot_line(
        "Delay Trend Comparison",
        "Delay (ms)",
        methods,
        [results[m]["Delay"] for m in methods],
        "trend_delay_comparison.png",
    )

    # Energy trend
    plot_line(
        "Energy Consumption Trend",
        "Energy (mJ)",
        methods,
        [results[m]["Energy"] for m in methods],
        "trend_energy_comparison.png",
    )

    # PDR trend
    plot_line(
        "Packet Delivery Ratio Trend",
        "PDR",
        methods,
        [results[m]["PDR"] for m in methods],
        "trend_pdr_comparison.png",
    )

    # Overall QoS trend
    plot_overall_trend(methods, results)

    print("Visualization trend figures generated successfully in ./out/")
