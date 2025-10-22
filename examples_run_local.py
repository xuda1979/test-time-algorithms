#!/usr/bin/env python3
# examples_run_local.py
# Local, no-internet demonstrations of the verification components used by the pro scripts.
import json, time, random, math, sys, os

try:
    import sympy as sp
    from sympy.physics import units as U
    HAVE_SYMPY = True
except Exception:
    HAVE_SYMPY = False

def math_conjecture_tests(K=600):
    """
    Evaluate a set of conjectures on random integer samples.
    Returns dict with pass rates and first counterexamples.
    """
    results = []
    rng = random.Random(12345)

    # Domain for n: 1..500
    def sample_n():
        if rng.random() < 0.85:
            return rng.randint(1, 60)
        return rng.randint(61, 500)

    # Conjecture 1 (True): sum_{k=1}^n k^3 = (n(n+1)/2)^2
    def c1(n):
        s = n*(n+1)//2
        lhs = sum(k**3 for k in range(1, n+1))
        rhs = s*s
        return lhs == rhs

    # Conjecture 2 (True): n^2 + n is even for all n>=1
    def c2(n):
        return ((n*n + n) % 2) == 0

    # Conjecture 3 (True): n^2 ≡ n (mod 2)
    def c3(n):
        return ((n*n - n) % 2) == 0

    # Conjecture 4 (False): n^2 + n + 1 is prime for all n>=1
    def is_prime(x):
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        r = int(x**0.5)
        f = 3
        while f <= r:
            if x % f == 0:
                return False
            f += 2
        return True

    def c4(n):
        return is_prime(n*n + n + 1)

    tests = [
        ("C1: sum k^3 identity", c1),
        ("C2: n^2+n even", c2),
        ("C3: n^2 ≡ n (mod 2)", c3),
        ("C4: n^2+n+1 is prime", c4),
    ]

    for name, fn in tests:
        ok, tried, first_cex = 0, 0, None
        for _ in range(K):
            n = sample_n()
            tried += 1
            holds = False
            try:
                holds = bool(fn(n))
            except Exception:
                holds = False
            if holds:
                ok += 1
            elif first_cex is None:
                first_cex = {"n": n}
        results.append({
            "name": name,
            "K": K,
            "passrate": ok / max(1, tried),
            "first_counterexample": first_cex
        })
    return {"domain": "math", "results": results}

def algorithm_max_subarray_len_range(num_cases=300):
    """
    Compare a correct O(n) candidate against a brute-force oracle
    for maximum subarray sum where length is between L and R inclusive.
    """
    rng = random.Random(12321)

    def gen_case():
        n = rng.randint(5, 60)
        arr = [rng.randint(-50, 60) for _ in range(n)]
        L = rng.randint(1, min(8, n))
        R = rng.randint(L, min(12, n))
        return arr, L, R

    def brute(arr, L, R):
        best = -10**18
        n = len(arr)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += arr[j]
                m = j - i + 1
                if L <= m <= R:
                    if s > best:
                        best = s
        return best

    from collections import deque
    def candidate(arr, L, R):
        n = len(arr)
        pref = [0]*(n+1)
        for i in range(1, n+1):
            pref[i] = pref[i-1] + arr[i-1]
        best = -10**18
        dq = deque()
        for i in range(1, n+1):
            idx_add = i - L
            if idx_add >= 0:
                while dq and pref[dq[-1]] >= pref[idx_add]:
                    dq.pop()
                dq.append(idx_add)
            idx_min = i - R
            while dq and dq[0] < idx_min:
                dq.popleft()
            if dq:
                best = max(best, pref[i] - pref[dq[0]])
        return best

    ok, total = 0, 0
    t0 = time.perf_counter()
    failures = []
    for _ in range(num_cases):
        arr, L, R = gen_case()
        b = brute(arr, L, R)
        c = candidate(arr, L, R)
        total += 1
        if b == c:
            ok += 1
        else:
            failures.append({"arr": arr, "L": L, "R": R, "brute": b, "candidate": c})
            if len(failures) > 5:
                failures = failures[-5:]
    dt = time.perf_counter() - t0
    return {"domain": "algorithms", "acc": ok / max(1,total), "n": total, "time_sec": dt, "failures": failures}

def physics_dimensional_checks():
    """
    Check dimensional homogeneity for a few toy hypotheses.
    Returns a summary of results.
    """
    if not HAVE_SYMPY:
        return {"domain": "theoretical_physics", "note": "sympy not available", "results": []}

    import sympy as sp
    from sympy.physics import units as U

    # simple unit dimension function
    M = (1,0,0); L=(0,1,0); T=(0,0,1)
    def vadd(a,b): return tuple(x+y for x,y in zip(a,b))
    def vscale(a,k): return tuple(x*k for x in a)
    UNIT_DIM = {
        U.kilogram: M,
        U.meter: L,
        U.second: T,
        U.newton: vadd(M, vadd(L, vscale(T,-2))),
        U.joule: vadd(vadd(M, vscale(L,2)), vscale(T,-2)),
    }
    def dim_expr(expr):
        if expr in UNIT_DIM:
            return UNIT_DIM[expr]
        if isinstance(expr, sp.Symbol):
            return (0,0,0)
        if isinstance(expr, (sp.Integer, sp.Float)):
            return (0,0,0)
        if isinstance(expr, sp.Mul):
            d=(0,0,0)
            for a in expr.args:
                d=vadd(d, dim_expr(a))
            return d
        if isinstance(expr, sp.Pow):
            base, exp = expr.as_base_exp()
            if exp.is_Number:
                return vscale(dim_expr(base), int(exp))
            return (0,0,0)
        if isinstance(expr, sp.Add):
            dims = [dim_expr(a) for a in expr.args]
            ok = all(d==dims[0] for d in dims)
            return dims[0] if ok else ("MISMATCH",)
        return (0,0,0)
    def dim_equal(lhs, rhs): return dim_expr(lhs) == dim_expr(rhs)

    m, a, F = sp.symbols('m a F')
    v, E, p = sp.symbols('v E p')
    hyps = [
        ("H1: Newton's second law", sp.Eq(F, m*a), {m: U.kilogram, a: U.meter/U.second**2, F: U.newton}),
        ("H2: Kinetic energy (wrong coefficient but dimensionally OK)", sp.Eq(E, m*v**2), {m: U.kilogram, v: U.meter/U.second, E: U.joule}),
        ("H3: Bad momentum relation", sp.Eq(p, m + v), {m: U.kilogram, v: U.meter/U.second, p: U.kilogram*U.meter/U.second}),
    ]
    rows = []
    for name, eq, units in hyps:
        lhs = eq.lhs.subs(units)
        rhs = eq.rhs.subs(units)
        rows.append({"name": name, "dim_ok": bool(dim_equal(lhs, rhs))})
    return {"domain": "theoretical_physics", "results": rows}

def coding_tiny_example():
    """
    Local coding demo: implement a tiny tabular query (filter + sum) over a list of dicts.
    Simulates 'code passes tests' metric without external deps.
    """
    data = [
        {"city": "A", "val": 10, "type": "x"},
        {"city": "B", "val": 5,  "type": "y"},
        {"city": "A", "val": 7,  "type": "x"},
        {"city": "A", "val": -3, "type": "z"},
        {"city": "B", "val": 2,  "type": "x"},
    ]
    def query_sum(rows, where):
        return sum(r["val"] for r in rows if where(r))
    passed = 0; total = 0
    total += 1; passed += int(query_sum(data, lambda r: r["city"]=="A") == 14)
    total += 1; passed += int(query_sum(data, lambda r: r["type"]=="x") == 19)
    total += 1; passed += int(query_sum(data, lambda r: r["city"]=="C") == 0)
    total += 1; passed += int(query_sum(data, lambda r: r["type"]!="z") == 24)
    return {"domain": "coding", "tests_passed": passed, "tests_total": total}

def main():
    out = []
    out.append(math_conjecture_tests(K=600))
    out.append(algorithm_max_subarray_len_range(num_cases=300))
    out.append(physics_dimensional_checks())
    out.append(coding_tiny_example())
    with open("experiment_results.json","w",encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # generate results_tables.tex
    rows_math = out[0]["results"]
    math_table = ["\\begin{tabular}{lccc}", "\\toprule",
                  "Conjecture & K & Pass rate & First counterexample \\\\ \\midrule"]
    for r in rows_math:
        fc = r["first_counterexample"]
        fc_str = f"{fc}" if fc else "-"
        math_table.append(f"{r['name']} & {r['K']} & {r['passrate']:.3f} & {fc_str} \\\")
    math_table += ["\\bottomrule", "\\end{tabular}"]

    algo = out[1]
    algo_table = ["\\begin{tabular}{lccc}", "\\toprule",
                  "Metric & Value & Note & Failures \\\\ \\midrule"]
    algo_table.append(f"Accuracy vs. oracle & {algo['acc']:.3f} & {algo['n']} cases & {len(algo['failures'])} \\\")
    algo_table.append(f"Wall time (s) & {algo['time_sec']:.3f} & Python local & - \\\")
    algo_table += ["\\bottomrule", "\\end{tabular}"]

    phys_rows = out[2]["results"]
    phys_table = ["\\begin{tabular}{lc}", "\\toprule", "Hypothesis & Dim OK \\\\ \\midrule"]
    for r in phys_rows:
        phys_table.append(f"{r['name']} & {'Yes' if r['dim_ok'] else 'No'} \\\")
    phys_table.append(f"{r['name']} & {'Yes' if r['dim_ok'] else 'No'} \\\")
    phys_table += ["\\bottomrule", "\\end{tabular}"]

    code = out[3]
    code_table = ["\\begin{tabular}{lc}", "\\toprule", "Metric & Value \\\\ \\midrule"]
    code_table.append(f"Unit tests passed / total & {code['tests_passed']} / {code['tests_total']} \\\")
    code_table += ["\\bottomrule", "\\end{tabular}"]

    tex = r"""
% results_tables.tex (auto-generated)
\section{Experimental Results (Local Demonstrations)}
This section reports local, offline demonstrations of the verification components.

\subsection{Mathematics}
\noindent
""" + "\n".join(math_table) + r"""

\subsection{Algorithms}
\noindent
""" + "\n".join(algo_table) + r"""

\subsection{Theoretical Physics}
\noindent
""" + "\n".join(phys_table) + r"""

\subsection{Coding}
\noindent
""" + "\n".join(code_table) + r"""

\paragraph{Notes.} These runs were produced locally without calling external APIs.
They exercise the verification backends used by the full LLM pipelines.
"""

    with open("results_tables.tex","w",encoding="utf-8") as f:
        f.write(tex)

if __name__ == "__main__":
    main()
