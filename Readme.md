# Currency Arbitrage Optimization using QAOA in Cirq  
### A Comparative Study with Classical Ising Solvers

## Overview

This repository contains a research-oriented implementation of **currency arbitrage detection** formulated as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem and solved using both **classical Ising-based optimization methods** and a **quantum variational approach** based on the **Quantum Approximate Optimization Algorithm (QAOA)** implemented in **Google Cirq**.

The project is a **direct implementation and comparative study** based on the reference work:

*S. Deshpande, E. Das, and F. Mueller,  
“Currency Arbitrage Optimization using Quantum Annealing, QAOA and Constraint Mapping,”  
arXiv preprint arXiv:2502.15742v1, 2025.*

**Please Note for a complete and elaborated technical description, mathematical formulation, experimental results, and discussion, kindly refer to the project report provided in the uploaded PDF.**

---

## Motivation

Classical algorithms can efficiently detect arbitrage opportunities in small currency networks, but their computational complexity increases rapidly as the number of currencies grows. Prior quantum implementations of QAOA-particularly using Qiskit-have demonstrated feasibility but often suffered from constraint violations and parameter instability.

This project explores whether a **Cirq-based QAOA implementation** can model the arbitrage problem more faithfully and how its performance compares against classical Ising solvers **under an identical QUBO formulation**.

---

## Key Focus

- QUBO formulation of currency arbitrage with explicit cycle constraints  
- Classical baselines: simulated annealing, brute-force enumeration, and gradient-based QUBO minimization  
- Cirq-based QAOA with explicit Hamiltonian construction  
- Direct classical–quantum comparison on the same optimization model  
- Emphasis on correctness, constraint satisfaction, and interpretability  

---

## Methodology Summary

- Currencies are modeled as nodes in a directed graph  
- Exchange rates define weighted directed edges  
- Arbitrage detection is reduced to minimizing a QUBO cost function  
- Constraints are enforced via quadratic penalty terms  
- Classical Ising solvers validate optimality  
- QAOA is used to reproduce the same solution structure through Hamiltonian-based optimization  

---

## Results Summary

- Classical solvers reliably find the global optimum for small problem sizes  
- Cirq-based QAOA reproduces the same low-energy, constraint-feasible solutions  
- Valid arbitrage cycles correspond to minimum-energy states  
- Constraint-violating configurations are naturally penalized  
- Cirq provides improved modularity and interpretability compared to earlier Qiskit-based approaches  

---

## References

[1] S. Deshpande, E. Das, and F. Mueller,  
“Currency Arbitrage Optimization using Quantum Annealing, QAOA and Constraint Mapping,”  
arXiv preprint arXiv:2502.15742v1, 2025.

[2] E. Farhi, J. Goldstone, and S. Gutmann,  
“A Quantum Approximate Optimization Algorithm,”  
arXiv preprint arXiv:1411.4028, 2014.

[3] Google Quantum AI,  
“Cirq Documentation,”  
https://quantumai.google/cirq

[4] F. Glover, G. Kochenberger, and Y. Du,  
“Quantum Bridge Analytics I: A Tutorial on Formulating and Using QUBO Models,”  
arXiv preprint arXiv:1811.11538, 2019.

[5] K. Tatsumura, R. Hidaka, M. Yamasaki, Y. Sakai, and H. Goto,  
“A Currency Arbitrage Machine Based on the Simulated Bifurcation Algorithm for Ultrafast Detection of Optimal Opportunity,”  
IEEE International Symposium on Circuits and Systems (ISCAS), 2020.

[6] G. Carrascal, B. Roman, A. Barrio, and G. Botella,  
“Differential Evolution VQE for Crypto-Currency Arbitrage,”  
Digital Signal Processing, vol. 148, 2024.
