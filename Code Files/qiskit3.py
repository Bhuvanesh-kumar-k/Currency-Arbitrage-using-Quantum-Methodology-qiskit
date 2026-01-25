
"""
QC_Project_Currency_Arbitrage_QAOA
---------------------------------
This script demonstrates Optimal Currency Arbitrage Detection using Qiskit.
It loads currency exchange rates from a CSV file, builds a graph, formulates
a QUBO optimization problem, solves it with QAOA, compares with classical
algorithms, and visualizes arbitrage cycles.
"""

# -------------------------------
# 1. Install required libraries
# -------------------------------
# Run these commands in terminal if not already installed:
# python -m pip install qiskit qiskit-aer qiskit-optimization matplotlib networkx pandas qiskit-algorithms

# -------------------------------
# 2. Import required libraries
# -------------------------------
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
import sys
import os

# Qiskit imports
from qiskit_aer import Aer
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

# -------------------------------
# 3. Load and validate dataset
# -------------------------------
csv_path = r"D:\Windsurf\CA\Currency-Arbitrage-using-Quantum-Methodology-qiskit\data\Data-sheet.csv"

# Check if file exists
if not os.path.exists(csv_path):
    print(f"Error: File not found at {csv_path}")
    sys.exit(1)

# Load CSV
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    sys.exit(1)

# Validate required columns
required_columns = {"Source", "Target", "Rate"}
if not required_columns.issubset(df.columns):
    print(f"Error: CSV must contain columns: {required_columns}")
    sys.exit(1)

# Check for missing values
if df[list(required_columns)].isnull().any().any():
    print("Error: CSV contains missing values in required columns.")
    sys.exit(1)

# Check for non-numeric rates
if not pd.api.types.is_numeric_dtype(df["Rate"]):
    print("Error: 'Rate' column must be numeric.")
    sys.exit(1)

print("✅ Dataset loaded and validated successfully!")
print(df.head())

# -------------------------------
# 4. Build directed currency graph
# -------------------------------
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row["Source"], row["Target"], weight=row["Rate"])

print("\nCurrencies in dataset:", list(G.nodes))
print("Number of exchange rates:", G.number_of_edges())

# Visualize the graph
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Currency Exchange Graph")
plt.show()

# -------------------------------
# 5. Formulate QUBO problem
# -------------------------------
qp = QuadraticProgram()

# Add binary variables for each edge
edge_list = list(G.edges())
for i in range(len(edge_list)):
    qp.binary_var(name=f"x{i}")

# Build objective: minimize negative log of exchange rates
linear_coeffs = {}
for i, (src, dst) in enumerate(edge_list):
    rate = G[src][dst]['weight']
    linear_coeffs[f"x{i}"] = -math.log(rate)

qp.minimize(linear=linear_coeffs)

print("\nQUBO problem formulated with", qp.get_num_vars(), "variables.")

# -------------------------------
# 6. Solve with QAOA
# -------------------------------
qaoa = QAOA(optimizer=COBYLA(maxiter=100), reps=2,
            quantum_instance=Aer.get_backend('qasm_simulator'))
optimizer = MinimumEigenOptimizer(qaoa)

result = optimizer.solve(qp)
print("\n✅ QAOA solution:")
print(result)

# -------------------------------
# 7. Classical comparison
# -------------------------------
log_graph = nx.DiGraph()
for (src, dst), data in G.edges.items():
    rate = data['weight']
    log_graph.add_edge(src, dst, weight=-math.log(rate))

try:
    cycle = nx.find_negative_cycle(log_graph, source=list(G.nodes)[0])
    print("\n🔍 Classical Bellman-Ford arbitrage cycle:", cycle)
except Exception as e:
    print("\nNo arbitrage cycle detected classically.")

# -------------------------------
# 8. Performance metrics
# -------------------------------
num_qubits = qp.get_num_vars()
print("\n📊 Performance Metrics:")
print("  Number of qubits:", num_qubits)
print("  QAOA reps (layers):", qaoa.reps)
print("  Optimizer:", type(qaoa.optimizer).__name__)

# -------------------------------
# 9. Visualization of QAOA solution
# -------------------------------
selected_edges = []
for i, edge in enumerate(edge_list):
    if result.x[i] > 0.5:
        selected_edges.append(edge)

print("\n🔗 Selected edges from QAOA:", selected_edges)

edge_colors = ["red" if edge in selected_edges else "black" for edge in G.edges()]
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000,
        edge_color=edge_colors)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Arbitrage Cycle Highlighted (QAOA)")
plt.show()
