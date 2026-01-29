"""
Enhanced Qiskit Currency Arbitrage Detection with Full Terminal Output
-----------------------------------------------------------------------
This script replicates the detailed outputs of the Cirq notebook using Qiskit.
It includes QUBO formulation, Ising conversion, QAOA circuit, simulation,
bitstring analysis, energy calculations, and arbitrage path validation.
"""


# -------------------------------
# 1. Install required libraries
# -------------------------------
# Run these commands in terminal if not already installed:
# python -m pip install "qiskit==0.39.2" qiskit-aer "qiskit-optimization==0.4.0" matplotlib networkx pandas qiskit-algorithms

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
import os
import sys

from qiskit_aer import Aer
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.converters import QuadraticProgramToQubo

# -------------------------------
# 1. Load and validate dataset
# -------------------------------
csv_path = r"D:\Windsurf\CA\Currency-Arbitrage-using-Quantum-Methodology-qiskit\data\Data-sheet.csv"

if not os.path.exists(csv_path):
    print(f"Error: File not found at {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)
required_columns = ["Source", "Target", "Rate"]
if not all(col in df.columns for col in required_columns):
    print(f"Error: CSV must contain columns: {required_columns}")
    sys.exit(1)

if df[required_columns].isnull().any().any():
    print("Error: CSV contains missing values.")
    sys.exit(1)

if not pd.api.types.is_numeric_dtype(df["Rate"]):
    print("Error: 'Rate' column must be numeric.")
    sys.exit(1)

print("✅ Dataset loaded successfully!")
print(df.head())

# -------------------------------
# 2. Build graph and map qubits
# -------------------------------
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row["Source"], row["Target"], weight=row["Rate"])

currencies = list(G.nodes)
num_currencies = len(currencies)
edges = list(G.edges())
num_variables = len(edges)

print(f"Currencies: {currencies}")
print(f"Number of decision variables (edges): {num_variables}")

print("Qubit-to-Edge Mapping:")
for i, (src, dst) in enumerate(edges):
    print(f"  Qubit {i}: b_{{{src}->{dst}}}")

# -------------------------------
# 2a. Visualize the currency graph
# -------------------------------
pos = nx.spring_layout(G, seed=42)  # consistent layout
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Currency Exchange Graph")
plt.show()


# -------------------------------
# 3. Define QUBO coefficients
# -------------------------------
penalty_factor = 5.0
linear = {}
quadratic = {}

print("Linear Coefficients (Profit Terms):")
for i, (src, dst) in enumerate(edges):
    rate = G[src][dst]["weight"]
    coeff = -math.log(rate)
    linear[f"x{i}"] = coeff
    print(f"  x{i} ({src}->{dst}): {coeff:.6f}")

# Penalty terms
print("Adding Penalty Terms:")
for i in range(num_variables):
    linear[f"x{i}"] -= 4 * penalty_factor

# Outgoing and incoming constraints
for src in currencies:
    out_vars = [f"x{j}" for j, (u, v) in enumerate(edges) if u == src]
    for a in range(len(out_vars)):
        for b in range(a + 1, len(out_vars)):
            quadratic[(out_vars[a], out_vars[b])] = penalty_factor

for dst in currencies:
    in_vars = [f"x{j}" for j, (u, v) in enumerate(edges) if v == dst]
    for a in range(len(in_vars)):
        for b in range(a + 1, len(in_vars)):
            quadratic[(in_vars[a], in_vars[b])] = penalty_factor

constant = 2 * num_currencies * penalty_factor
print(f"Penalty factor: {penalty_factor}, Constant term: {constant}")
print(f"Number of quadratic terms: {len(quadratic)}")

# -------------------------------
# 4. Formulate QUBO
# -------------------------------
qp = QuadraticProgram()
for i in range(num_variables):
    qp.binary_var(name=f"x{i}")
qp.minimize(linear=linear, quadratic=quadratic)
print("QUBO Problem:")
print(qp.export_as_lp_string())

# -------------------------------
# 5. Convert to Ising
# -------------------------------
# Suppose qp is your QuadraticProgram
qubo_converter = QuadraticProgramToQubo()
qubo = qubo_converter.convert(qp)

# Get Ising operator
ising_op, offset = qubo.to_ising()

print("Ising Hamiltonian:")
print("Ising operator:", ising_op)
print("Offset:", offset)

print(ising_op)
print(f"Ising constant offset: {offset:.4f}")

# -------------------------------
# 6. QAOA Circuit and Simulation
# -------------------------------
backend = Aer.get_backend("qasm_simulator")
qaoa = QAOA(optimizer=COBYLA(maxiter=100), reps=2, backend=backend)

optimizer = MinimumEigenOptimizer(qaoa)

result = optimizer.solve(qubo)
print("✅ QAOA Result:")
print(result)

# -------------------------------
# 7. Analyze Bitstring
# -------------------------------
bitstring = "".join(str(int(x)) for x in result.x)
print(f"Most frequent bitstring: {bitstring}")
selected_edges = [edges[i] for i, bit in enumerate(result.x) if bit > 0.5]
print(f"Selected edges: {selected_edges}")

# -------------------------------
# 8. Validate Cycle
# -------------------------------
def is_valid_cycle(edges, currencies):
    if len(edges) != len(currencies):
        return False, "Incorrect number of edges"
    incoming = {c: 0 for c in currencies}
    outgoing = {c: 0 for c in currencies}
    for u, v in edges:
        outgoing[u] += 1
        incoming[v] += 1
    for c in currencies:
        if incoming[c] != 1 or outgoing[c] != 1:
            return False, f"Currency {c}: {incoming[c]} in, {outgoing[c]} out"
    return True, "Valid cycle"

valid, msg = is_valid_cycle(selected_edges, currencies)
print(f"Cycle Validation:")
print(f"  Is valid cycle? {valid}")
print(f"  Details: {msg}")