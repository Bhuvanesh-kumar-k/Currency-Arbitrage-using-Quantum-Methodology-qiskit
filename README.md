# Currency Arbitrage using QAOA (Qiskit)

## Overview
This project models **currency arbitrage detection** as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem and solves it using the **Quantum Approximate Optimization Algorithm (QAOA)** implemented in **Qiskit**.

Currencies are represented as nodes in a directed graph and exchange rates as directed, weighted edges. The goal is to find a cycle whose product of rates is greater than 1 (profitable arbitrage). The multiplicative objective is converted into an additive one using the transformation `-log(rate)`.

## Authors
- Saranya S (2577421)
- Bhuvanesh Kumar (1844341)
- Santhossh J U (1982102)
- Mohammad Yaseen (2160066)
- Gowthaman A (1969311)


## Repository structure
- `Code Files/`
  - `Currency_Arbitrage_QAOA_Qiskit.ipynb` — main notebook implementation (QUBO → Ising → QAOA + validation)
  - `Qiskit.py` — script version of the workflow
- `Data/`
  - `Data-sheet.csv` — exchange-rate dataset used to build the graph
- `QAOA_Qiskit_Arbitrage_Report.pdf` — report/exported results
- `ProjectDescription.txt` — short project summary

## Methodology (high level)
- **Graph modeling**
  - Nodes: currencies (USD, EUR, …)
  - Directed edges: conversions with rate `r_ij`

- **QUBO formulation**
  - Binary variable per directed edge: `x_e ∈ {0,1}`
  - Objective (profit term): minimize `Σ (-log(r_e)) x_e`
  - Constraints encoded using penalties:
    - each currency has exactly **one outgoing** selected edge
    - each currency has exactly **one incoming** selected edge

- **Quantum execution (QAOA)**
  - Convert QUBO → **Ising Hamiltonian**
  - Run QAOA (simulated) to sample bitstrings
  - Decode bitstrings back to chosen edges (trades)

- **Validation and interpretation**
  - Feasibility checks (cycle constraints)
  - Classical baseline: Bellman–Ford negative-cycle detection on `-log(rate)`
  - Profit computation + fee sensitivity analysis

## How to run

### Option A — Jupyter Notebook (recommended)
1. Open:
   - `Code Files/Currency_Arbitrage_QAOA_Qiskit.ipynb`
2. Ensure the dataset exists at:
   - `Data/Data-sheet.csv`
3. Run cells top-to-bottom.

### Option B — Python script
Run:
```bash
python "Code Files/Qiskit.py"
```
Notes:
- The script currently contains an **absolute `csv_path`**; update it if your local path differs.
- The repository uses `Data/Data-sheet.csv` (capital `D`). If your environment is case-sensitive, ensure the script points to the correct folder name.

## Dependencies
The notebook was executed using Python 3.10 and these common packages:
- `qiskit`, `qiskit-aer`, `qiskit-optimization`, `qiskit-algorithms`
- `numpy`, `pandas`, `networkx`, `matplotlib`

If you want a quick setup:
```bash
pip install -U pip setuptools wheel
pip install qiskit qiskit-aer qiskit-optimization qiskit-algorithms numpy pandas networkx matplotlib
```

## Expected outputs
When you run the notebook/script, you will typically see:
- A plotted exchange-rate directed graph
- A printed QUBO (linear/quadratic terms) and Ising Hamiltonian
- QAOA sampling results (bitstrings + decoded selected edges)
- Classical baseline detection (Bellman–Ford) and profit computation
- Fee sensitivity (break-even fee estimate)

## Notes
- This is a research/demo-style project. Real-world arbitrage depends on spreads, fees, slippage, and latency.
- QAOA is executed on simulators in this repo; running on hardware would require additional configuration.
