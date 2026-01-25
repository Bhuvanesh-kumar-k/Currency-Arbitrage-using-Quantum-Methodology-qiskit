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

## python -m pip install qiskit qiskit-aer qiskit-optimization matplotlib networkx pandas qiskit-algorithms

## python -m pip install "qiskit==0.39.2"
## python -m pip install matplotlib
## python -m pip install pandas
## python -m pip install networkx
## python -m pip install qiskit
## python -m pip install qiskit-optimization
## python -m pip install qiskit-aer
## python -m pip install qiskit-algorithms






# -------------------------------
# 1. Import required libraries
# -------------------------------
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math

# Qiskit imports
from qiskit_aer import Aer
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA


# -------------------------------
# 2. Load dataset from CSV
# -------------------------------
# Path to your dataset
csv_path = r"D:\Windsurf\CA\Currency-Arbitrage-using-Quantum-Methodology-qiskit\data\Data-sheet.csv"

# Read CSV file (columns: Source, Target, Rate)
df = pd.read_csv(csv_path)

print("Dataset loaded successfully!")
print(df.head())

# -------------------------------
# 3. Build directed currency graph
# -------------------------------
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row["Source"], row["Target"], weight=row["Rate"])

print("\nCurrencies in dataset:", list(G.nodes))
print("Number of exchange rates:", G.number_of_edges())

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Currency Exchange Graph")
plt.show()

# -------------------------------
# 4. Formulate QUBO problem
# -------------------------------
# QuadraticProgram is used to define optimization problems for QAOA
qp = QuadraticProgram()

# Add binary variables for each edge (1 = include in cycle, 0 = exclude)
for i, edge in enumerate(G.edges()):
    qp.binary_var(name=f"x{i}")

# Objective: maximize product of rates → equivalently minimize negative log
objective = 0
for i, (src, dst) in enumerate(G.edges()):
    rate = G[src][dst]['weight']
    objective += -math.log(rate) * qp.variables[i]

qp.minimize(linear=objective)

print("\nQUBO problem formulated with", qp.get_num_vars(), "variables.")

# -------------------------------
# 5. Solve with QAOA
# -------------------------------
# Define QAOA algorithm with COBYLA optimizer
qaoa = QAOA(optimizer=COBYLA(maxiter=100), reps=2,
            quantum_instance=Aer.get_backend('qasm_simulator'))
optimizer = MinimumEigenOptimizer(qaoa)

# Solve the QUBO
result = optimizer.solve(qp)
print("\nQAOA solution:")
print(result)

# -------------------------------
# 6. Classical comparison
# -------------------------------
# Use Bellman-Ford to detect arbitrage (negative cycle in log graph)
log_graph = nx.DiGraph()
for (src, dst), data in G.edges.items():
    rate = data['weight']
    log_graph.add_edge(src, dst, weight=-math.log(rate))

try:
    cycle = nx.find_negative_cycle(log_graph, source=list(G.nodes)[0])
    print("\nClassical Bellman-Ford arbitrage cycle:", cycle)
except Exception as e:
    print("\nNo arbitrage cycle detected classically.")

# -------------------------------
# 7. Performance metrics
# -------------------------------
# Track qubit count, circuit depth, execution time
num_qubits = qp.get_num_vars()
print("\nPerformance Metrics:")
print("  Number of qubits:", num_qubits)
print("  QAOA reps (layers):", qaoa.reps)
print("  Optimizer:", type(qaoa.optimizer).__name__)

# -------------------------------
# 8. Visualization of solution
# -------------------------------
# Highlight selected edges from QAOA solution
selected_edges = []
for i, edge in enumerate(G.edges()):
    if result.x[i] > 0.5:  # variable chosen
        selected_edges.append(edge)

print("\nSelected edges from QAOA:", selected_edges)

edge_colors = ["red" if edge in selected_edges else "black" for edge in G.edges()]
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000,
        edge_color=edge_colors)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Arbitrage Cycle Highlighted (QAOA)")
plt.show()
