import os
import pickle
import networkx as nx

print("RUNNING FILE:", os.path.abspath(__file__))

with open("./out/iot_graph.gpickle", "rb") as f:
    G = pickle.load(f)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

print("\nSample nodes:")
for n, d in list(G.nodes(data=True))[:3]:
    print(n, d)

print("\nSample edges:")
for u, v, d in list(G.edges(data=True))[:3]:
    print(u, v, d)
