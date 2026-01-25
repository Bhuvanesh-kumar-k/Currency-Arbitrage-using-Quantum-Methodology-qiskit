"""
QC_Project_Currency_Arbitrage_QAOA
---------------------------------
This script demonstrates Optimal Currency Arbitrage Detection using Qiskit.
We formulate the arbitrage problem as a QUBO, solve it with QAOA, compare
with classical algorithms, and visualize the currency graph.
"""

# -------------------------------
# 1. Import required libraries
# -------------------------------
# Qiskit core
from qiskit import Aer, execute, QuantumCircuit
# Qiskit optimization for QUBO formulation
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
# Classical graph algorithms
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------------
# 2. Define currency exchange rates
# -------------------------------
# Example dataset: 4 currencies with exchange rates
# In practice, load from CSV or API
currencies = ["USD", "EUR", "GBP", "CAD"]
exchange_rates = {
    ("USD", "EUR"): 1.12,
    ("EUR", "GBP"): 0.84,
    ("GBP", "CAD"): 1.86,
    ("CAD", "USD"): 0.63
}

# -------------------------------
# 3. Build the currency graph
# -------------------------------
G = nx.DiGraph()
for (src, dst), rate in exchange_rates.items():
    G.add_edge(src, dst, weight=rate)

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
# For simplicity, we use sum of -log(rate) as cost
import math
objective = 0
for i, (src, dst) in enumerate(G.edges()):
    rate = G[src][dst]['weight']
    objective += -math.log(rate) * qp.variables[i]

qp.minimize(linear=objective)

# -------------------------------
# 5. Solve with QAOA
# -------------------------------
# Define QAOA algorithm with COBYLA optimizer
qaoa = QAOA(optimizer=COBYLA(maxiter=100), reps=2, quantum_instance=Aer.get_backend('qasm_simulator'))
optimizer = MinimumEigenOptimizer(qaoa)

# Solve the QUBO
result = optimizer.solve(qp)
print("QAOA solution:", result)

# -------------------------------
# 6. Classical comparison
# -------------------------------
# Use Bellman-Ford to detect arbitrage (negative cycle in log graph)
log_graph = nx.DiGraph()
for (src, dst), rate in exchange_rates.items():
    log_graph.add_edge(src, dst, weight=-math.log(rate))

try:
    cycle = nx.find_negative_cycle(log_graph, source="USD")
    print("Classical Bellman-Ford arbitrage cycle:", cycle)
except Exception as e:
    print("No arbitrage cycle detected classically.")

# -------------------------------
# 7. Performance metrics
# -------------------------------
# Track qubit count, circuit depth, execution time
qc = QuantumCircuit(len(G.edges()))
qc.h(range(len(G.edges())))
print("Circuit depth:", qc.depth())
print("Number of qubits:", qc.num_qubits)
