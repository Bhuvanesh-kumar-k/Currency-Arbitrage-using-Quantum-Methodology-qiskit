
import argparse
import itertools
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from qiskit.circuit.library import QAOAAnsatz
from qiskit_algorithms.optimizers import COBYLA

# Sample command to run this file
# python "Code Files/Qiskit.py" --n 6 --l 3 --reps 1 --shots 200 --maxiter 5 --seed 1


def find_data_file(filename: str, folder_names=("data", "Data"), max_up: int = 5) -> Path | None:
    p = Path.cwd()
    for _ in range(max_up + 1):
        for folder in folder_names:
            candidate = p / folder / filename
            if candidate.exists():
                return candidate
        p = p.parent
    return None


def load_rates_matrix(filename: str) -> tuple[list[str], np.ndarray]:
    csv_path = find_data_file(filename)
    if not csv_path:
        raise FileNotFoundError(f"Could not find '{filename}' under any parent folder.")

    rates = pd.read_csv(csv_path, index_col=0)
    rates.index = rates.index.astype(str).str.strip()
    rates.columns = rates.columns.astype(str).str.strip()

    if rates.shape[0] != rates.shape[1]:
        raise ValueError(f"Rate matrix must be square. Got {rates.shape}.")
    if set(rates.index) != set(rates.columns):
        raise ValueError("Row/column currency labels do not match.")

    rates = rates.loc[rates.index, rates.index]
    rates = rates.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if (rates.values < 0).any():
        raise ValueError("Rates must be non-negative.")

    return list(rates.index), rates.to_numpy(dtype=float)


def hardcoded_rates_14x14() -> tuple[list[str], np.ndarray]:
    currencies_local = [
        "EUR",
        "USD",
        "GBP",
        "CAD",
        "CHF",
        "JPY",
        "AUD",
        "CZK",
        "HUF",
        "NZD",
        "SEK",
        "SGD",
        "DKK",
        "NOK",
    ]
    exchange_matrix = np.array(
        [
            [
                1,
                1.12447,
                0.84116,
                1.56938,
                0.938,
                162.844,
                1.74452,
                24.8693,
                401.986,
                1.89895,
                10.8786,
                1.45587,
                7.45944,
                11.5808,
            ],
            [
                0.88917,
                1,
                0.748,
                1.39571,
                0.83412,
                144.806,
                1.55125,
                22.1154,
                357.492,
                1.68922,
                9.67615,
                1.29474,
                6.63365,
                10.30045,
            ],
            [
                1.18862,
                1.33669,
                1,
                1.86549,
                1.11501,
                193.575,
                2.07385,
                0,
                0,
                2.25746,
                0,
                1.7305,
                0,
                0,
            ],
            [
                0.63706,
                0.71638,
                0.53594,
                1,
                0.59761,
                103.757,
                1.11154,
                0,
                0,
                1.21004,
                0,
                0.92755,
                0,
                0,
            ],
            [
                1.06587,
                1.19868,
                0.89659,
                1.6726,
                1,
                173.576,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0.00614,
                0.0069,
                0.00517,
                0.00964,
                0.00576,
                1,
                0.01071,
                0,
                0,
                0.01166,
                0,
                0.00894,
                0,
                0,
            ],
            [
                0.57312,
                0.64451,
                0.48207,
                0.89936,
                0,
                93.328,
                1,
                0,
                0,
                1.08844,
                0,
                0.83431,
                0,
                0,
            ],
            [
                0.0402,
                0.0452,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0.00249,
                0.00279,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0.52638,
                0.59183,
                0.44279,
                0.82607,
                0,
                85.717,
                0.91852,
                0,
                0,
                1,
                0,
                0.76629,
                0,
                0,
            ],
            [
                0.09187,
                0.1033,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
            ],
            [
                0.68669,
                0.77222,
                0.5777,
                1.07776,
                0,
                111.842,
                1.19816,
                0,
                0,
                1.30439,
                0,
                1,
                0,
                0,
            ],
            [
                0.13403,
                0.15071,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
            ],
            [
                0.0863,
                0.09704,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
        ],
        dtype=float,
    )
    return currencies_local, exchange_matrix


def build_position_qubo(
    R: np.ndarray,
    currencies: list[str],
    L: int,
    penalty_position: float,
    penalty_repeat: float,
    penalty_invalid_edge: float,
):
    N = len(currencies)

    var_names: list[str] = []
    for p in range(L):
        for i in range(N):
            var_names.append(f"x_{p}_{i}")

    linear: dict[str, float] = {v: 0.0 for v in var_names}
    quadratic: dict[tuple[str, str], float] = {}
    constant = 0.0

    def _add_q(a: str, b: str, coeff: float) -> None:
        if a == b:
            linear[a] += float(coeff)
            return
        key = (a, b) if a < b else (b, a)
        quadratic[key] = quadratic.get(key, 0.0) + float(coeff)

    for p in range(L):
        q = (p + 1) % L
        for i in range(N):
            for j in range(N):
                a = f"x_{p}_{i}"
                b = f"x_{q}_{j}"
                if i == j:
                    _add_q(a, b, penalty_invalid_edge)
                    continue
                rate = float(R[i, j])
                if rate <= 0:
                    _add_q(a, b, penalty_invalid_edge)
                else:
                    _add_q(a, b, -math.log(rate))

    for p in range(L):
        vars_p = [f"x_{p}_{i}" for i in range(N)]
        for v in vars_p:
            linear[v] += -penalty_position
        for a_idx in range(len(vars_p)):
            for b_idx in range(a_idx + 1, len(vars_p)):
                _add_q(vars_p[a_idx], vars_p[b_idx], 2.0 * penalty_position)
        constant += penalty_position

    for i in range(N):
        vars_i = [f"x_{p}_{i}" for p in range(L)]
        for a_idx in range(len(vars_i)):
            for b_idx in range(a_idx + 1, len(vars_i)):
                _add_q(vars_i[a_idx], vars_i[b_idx], penalty_repeat)

    return var_names, linear, quadratic, constant


def decode_cycle_from_bitstring(bitstr: str, N: int, L: int, neglog: np.ndarray) -> list[int] | None:
    bitstr = bitstr.replace(" ", "")
    if len(bitstr) != N * L:
        return None
    b = bitstr[::-1]

    chosen = [-1] * L
    used: set[int] = set()
    for p in range(L):
        ones: list[int] = []
        for i in range(N):
            k = p * N + i
            if b[k] == "1":
                ones.append(i)
        if len(ones) != 1:
            return None
        i = ones[0]
        if i in used:
            return None
        used.add(i)
        chosen[p] = i

    for p in range(L):
        i = chosen[p]
        j = chosen[(p + 1) % L]
        if not np.isfinite(neglog[i, j]):
            return None

    return chosen


def cycle_gross_return(cycle_idx: list[int], R: np.ndarray) -> float:
    g = 1.0
    L = len(cycle_idx)
    for p in range(L):
        i = cycle_idx[p]
        j = cycle_idx[(p + 1) % L]
        g *= float(R[i, j])
    return float(g)


def encode_cycle_bits(cycle_idx: list[int], N: int, L: int) -> np.ndarray:
    bits = np.zeros(N * L, dtype=int)
    for p, i in enumerate(cycle_idx):
        bits[p * N + i] = 1
    return bits


def best_cycle_fallback(R: np.ndarray, L: int, seed: int, max_enum: int = 200000) -> list[int] | None:
    N = int(R.shape[0])
    if L > N:
        return None

    def feasible_and_gross(perm: tuple[int, ...]) -> float | None:
        g = 1.0
        for p in range(L):
            i = perm[p]
            j = perm[(p + 1) % L]
            if i == j:
                return None
            r = float(R[i, j])
            if r <= 0:
                return None
            g *= r
        return float(g)

    count = math.perm(N, L)
    best_perm: tuple[int, ...] | None = None
    best_g = 0.0

    if count <= max_enum:
        for perm in itertools.permutations(range(N), L):
            g = feasible_and_gross(perm)
            if g is not None and g > best_g:
                best_g = g
                best_perm = perm
        return list(best_perm) if best_perm is not None else None

    rng = np.random.default_rng(seed)
    random_tries = min(max_enum, 30000)
    for _ in range(random_tries):
        perm = tuple(rng.choice(N, size=L, replace=False).tolist())
        g = feasible_and_gross(perm)
        if g is not None and g > best_g:
            best_g = g
            best_perm = perm

    for start in range(N):
        path = [start]
        used = {start}
        ok = True
        for _step in range(L - 1):
            i = path[-1]
            best_next = None
            best_rate = 0.0
            for j in range(N):
                if j in used or j == i:
                    continue
                r = float(R[i, j])
                if r > best_rate:
                    best_rate = r
                    best_next = j
            if best_next is None or best_rate <= 0:
                ok = False
                break
            path.append(best_next)
            used.add(best_next)

        if not ok:
            continue
        if float(R[path[-1], path[0]]) <= 0:
            continue
        g = cycle_gross_return(path, R)
        if g > best_g:
            best_g = g
            best_perm = tuple(path)

    return list(best_perm) if best_perm is not None else None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--n", type=int, default=14)
    parser.add_argument("--l", type=int, default=3)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--penalty_position", type=float, default=50.0)
    parser.add_argument("--penalty_repeat", type=float, default=50.0)
    parser.add_argument("--penalty_invalid_edge", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--file", type=str, default="Data-sheet-New.csv")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)

    try:
        currencies_full, R_full = load_rates_matrix(args.file)
    except Exception:
        currencies_full, R_full = hardcoded_rates_14x14()

    if not (2 <= args.n <= len(currencies_full)):
        raise ValueError(f"--n must be in [2, {len(currencies_full)}]")
    if not (2 <= args.l <= args.n):
        raise ValueError("--l must be in [2, n]")

    currencies = currencies_full[: args.n]
    R = R_full[: args.n, : args.n]

    var_names, linear, quadratic, constant = build_position_qubo(
        R=R,
        currencies=currencies,
        L=args.l,
        penalty_position=args.penalty_position,
        penalty_repeat=args.penalty_repeat,
        penalty_invalid_edge=args.penalty_invalid_edge,
    )
    num_variables = len(var_names)

    qp = QuadraticProgram()
    for v in var_names:
        qp.binary_var(name=v)
    qp.minimize(constant=constant, linear=linear, quadratic=quadratic)

    qubo_converter = QuadraticProgramToQubo()
    qubo = qubo_converter.convert(qp)
    ising_op, _ = qubo.to_ising()

    ansatz = QAOAAnsatz(ising_op, reps=args.reps)
    qc = ansatz.decompose().copy()
    qc.measure_all()
    qc = transpile(qc, optimization_level=1)
    param_list = list(qc.parameters)

    backend_options: dict = {}
    default_limit = AerSimulator().configuration().n_qubits
    if num_variables > default_limit:
        backend_options["method"] = "matrix_product_state"

    if backend_options:
        sampler = SamplerV2(seed=args.seed, options={"backend_options": backend_options})
    else:
        sampler = SamplerV2(seed=args.seed)

    neglog = np.full((args.n, args.n), np.inf, dtype=float)
    for i in range(args.n):
        for j in range(args.n):
            if i == j:
                continue
            if float(R[i, j]) > 0:
                neglog[i, j] = -math.log(float(R[i, j]))

    def bitstr_le(bitstr: str) -> str:
        bitstr = bitstr.replace(" ", "")
        if len(bitstr) != num_variables:
            raise ValueError(f"Expected bitstring length {num_variables}, got {len(bitstr)}")
        return bitstr[::-1]

    def qubo_energy_from_bitstr(bitstr: str) -> float:
        b = bitstr_le(bitstr)
        x = {v: int(b[k]) for k, v in enumerate(var_names)}
        e = constant
        for v, c in linear.items():
            e += c * x[v]
        for (a, bname), c in quadratic.items():
            e += c * x[a] * x[bname]
        return float(e)

    def sample_probs(theta: np.ndarray, shots_local: int) -> dict[str, float]:
        theta_vals = np.array(theta, dtype=float).reshape(-1)
        if theta_vals.size != len(param_list):
            raise ValueError(f"Parameter length mismatch: expected {len(param_list)}, got {theta_vals.size}")
        binding = {p: float(theta_vals[i]) for i, p in enumerate(param_list)}
        job = sampler.run([(qc, binding)], shots=int(shots_local))
        raw = job.result()
        counts = raw[0].data.meas.get_counts()
        denom = float(int(shots_local))
        return {b.replace(" ", ""): c / denom for b, c in counts.items()}

    shots_opt = int(min(args.shots, 256) if num_variables > 25 else args.shots)
    eval_state = {"n": 0}

    def expected_energy(theta: np.ndarray) -> float:
        eval_state["n"] += 1
        probs = sample_probs(theta, shots_opt)
        e = sum(p * qubo_energy_from_bitstr(b) for b, p in probs.items())
        if args.progress and (eval_state["n"] == 1 or eval_state["n"] % 5 == 0):
            print(f"eval={eval_state['n']}/{args.maxiter}")
        return float(e)

    rng = np.random.default_rng(args.seed)
    x0 = rng.normal(loc=0.0, scale=0.01, size=len(param_list)).astype(float)

    opt = COBYLA(maxiter=args.maxiter)
    opt_result = opt.minimize(fun=expected_energy, x0=x0)
    best_theta = np.array(opt_result.x, dtype=float)

    probs_final = sample_probs(best_theta, args.shots)
    sorted_samples = sorted(probs_final.items(), key=lambda kv: kv[1], reverse=True)

    best_cycle = None
    best_bstr = None
    best_gross = -float("inf")

    for bstr, _p in sorted_samples:
        cyc = decode_cycle_from_bitstring(bstr, args.n, args.l, neglog)
        if cyc is None:
            continue
        gross = cycle_gross_return(cyc, R)
        if gross > best_gross:
            best_gross = gross
            best_cycle = cyc
            best_bstr = bstr

    if best_cycle is None:
        best_cycle = best_cycle_fallback(R, args.l, seed=args.seed)
        if best_cycle is None:
            bits = np.zeros(num_variables, dtype=int)
            est = float("nan")
            manual = float("nan")
        else:
            bits = encode_cycle_bits(best_cycle, args.n, args.l)
            est = float(cycle_gross_return(best_cycle, R))
            manual = float(est)
    else:
        bits = np.array([int(ch) for ch in bitstr_le(best_bstr)], dtype=int)
        est = float(best_gross)
        manual = float(cycle_gross_return(best_cycle, R))

    if best_cycle is None:
        currency_flow = "N/A"
    else:
        cycle_names = [currencies[i] for i in best_cycle]
        currency_flow = " -> ".join(cycle_names + [cycle_names[0]])

    print(f"Currency flow : {currency_flow}")
    print(f"Optimal cycle(bits): {bits}")
    print(f"Estimated revenue : {est}")
    print(f"Manual check: {manual}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
