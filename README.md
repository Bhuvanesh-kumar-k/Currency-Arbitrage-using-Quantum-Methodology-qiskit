# Currency Arbitrage using QAOA (Qiskit)

## Overview
This project models **currency arbitrage detection** as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem and solves it using the **Quantum Approximate Optimization Algorithm (QAOA)** implemented in **Qiskit**.

Exchange rates are loaded from a **square rate matrix** (row currency → column currency). A profitable arbitrage corresponds to a directed cycle whose product of rates is greater than 1. The multiplicative objective is converted into an additive one using the transformation `-log(rate)`.

## Authors
- Saranya Sundararajan (2577421)
- Bhuvanesh Kumar (1844341)
- Santhossh J U (1982102)
- Mohammad Yaseen (2160066)
- Gowthaman A (1969311)


## Repository structure
- `Code Files/`
  - `Currency_Arbitrage_QAOA_Qiskit.ipynb` — main notebook implementation (matrix → QUBO → Ising → QAOA + validation)
  - `Qiskit.py` — script version of the workflow
- `data/`
  - `Data-sheet-New.csv` — exchange-rate **matrix** dataset (recommended input)
  - `Data-sheet-New.txt` — same matrix dataset in tab-separated format
- `pdf.txt` — updated report text (you can paste this into Copilot/other tools to generate a PDF)
- `QAOA_Qiskit_Arbitrage_Report.pdf` — existing PDF report (legacy)

## Methodology (high level)
- **Graph modeling**
  - Nodes: currencies (USD, EUR, …)
  - Directed edges: conversions with rate `R[i,j]` (non-zero matrix entries)

- **QUBO formulation (position-based, fixed cycle length)**
  - Binary variable per (position, currency): `x[p,i] ∈ {0,1}`
  - Total variables / qubits: `N × L` where:
    - `N` = number of currencies
    - `L` = trading cycle length
  - Objective (profit term): minimize negative log-return along consecutive positions
  - Constraints encoded using penalties:
    - exactly one currency per position
    - no repeated currencies across positions
    - forbid invalid transitions where `R[i,j] = 0` (or `i=j`)

- **Quantum execution (QAOA)**
  - Convert QUBO → **Ising Hamiltonian**
  - Run QAOA (simulated) to sample bitstrings
  - Decode bitstrings back to a length-`L` currency cycle

- **Validation and interpretation**
  - Feasibility checks (cycle constraints)
  - Classical baseline: brute-force enumeration for small `(N, L)` settings
  - Profit computation + fee sensitivity analysis

## How to run

### Option A — Jupyter Notebook (recommended)
1. Open:
   - `Code Files/Currency_Arbitrage_QAOA_Qiskit.ipynb`
2. Ensure the dataset exists at:
   - `data/Data-sheet-New.csv`
3. Run cells top-to-bottom.

Notes:
- The QAOA execution cell prints an estimated remaining time (ETA) while COBYLA is running.
- Runtime depends strongly on `N × L`, `reps`, `shots`, and `maxiter`.

### Option B — Python script
Run:
```bash
python "Code Files/Qiskit.py" --n 6 --l 3 --reps 1 --shots 200 --maxiter 5 --seed 1"
```
Notes:
- The script currently contains an **absolute `csv_path`**; update it if your local path differs.
- The script is a legacy edge-list workflow and may not reflect the latest matrix-based notebook.

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
- QAOA sampling results (bitstrings + decoded cycles)
- Classical baseline enumeration (when feasible) and profit computation
- Fee sensitivity (break-even fee estimate)

## Notes
- This is a research/demo-style project. Real-world arbitrage depends on spreads, fees, slippage, and latency.
- QAOA is executed on simulators in this repo; running on hardware would require additional configuration.
