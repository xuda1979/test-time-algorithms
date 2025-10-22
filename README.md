# Test-Time Research Engines

This repository collects a family of **test-time research loops** built on OpenAI's
Responses API with **Structured Outputs**. They orchestrate planning, generation,
execution, repair, and scoring cycles across software engineering, theoretical
physics, algorithm design, and mathematics—and are now topped with a unified
`comprehensive_pro.py` orchestrator for mixed-domain studies.

All tools accept `--model gpt-4o`, `--model gpt-4o-mini`, `--model gpt-5`, or any
compatible Responses model that supports JSON schema structured outputs. Ensure
your environment exposes `OPENAI_API_KEY` and install the dependencies in
`requirements.txt`.

---

## 0. Quick prerequisites

```bash
# Core deps (adjust as needed)
pip install "openai>=1.40" rich sympy networkx python-dotenv

# Optional tools used by some flows
pip install ruff mypy pytest z3-solver

# API key
export OPENAI_API_KEY=sk-...
```

> Swap models freely, e.g. `--model gpt-4o`, `--model gpt-4o-mini`, or `--model gpt-5`
> (if available).

---

## 1. Thinking styles by domain（领域的“思维方式”）

The engines share seven reasoning paradigms—LinearCoT, Tree-of-Thought (ToT),
Graph-of-Thought (GoT), Reflexion, Multi-Agent, Meta-Cognitive planning, and
Self-Evolving repair—but emphasize different aspects per discipline.

### Coding（代码构造）

- **Spec → Design → Code → Test → Repair** loop anchored in precise contracts.
- Focus on **local reasoning** (invariants, pre/postconditions, types, effects).
- **Search operators**: template synthesis, test-driven repair, API minimization,
  latency/complexity trade-offs.
- Heavy reliance on tool feedback: linters, type checkers, property-based tests.

### Theoretical Physics（理论物理）

- **Model building** through symmetries, conservation laws, and effective theories.
- **Constraint-first** reasoning with dimensional analysis, limiting behavior,
  and invariance checks (scaling, parity, gauge where appropriate).
- Emphasis on falsification: compare against known regimes, toy models, and
  consistency conditions.

### Algorithm Discovery（算法发现）

- **Representation + invariant design** drives data structures and loop invariants.
- Explore **competing paradigms** (greedy, DP, divide-and-conquer, randomized,
  approximation).
- Pair each candidate with a **brute-force oracle** for correctness checks on
  small instances.
- Outline **proof skeletons**: exchange arguments, potentials, amortized bounds.

---

## 2. Fully worked examples & CLI recipes

Each script is standalone, mirrors the design of `math_discovery_pro.py`, and
relies on Structured Outputs for deterministic artifact handling. The snippets
below are copy/paste ready.

### 2.1 Mathematics discovery (`math_discovery_pro.py`)

**Additive number theory (binomial congruences)**

```bash
python math_discovery_pro.py \
  --model gpt-4o \
  --domain "additive number theory: binomial coefficients modulo small primes" \
  --rounds 2 --beam 24 --tests 600 --use-z3
```

**Combinatorial identities (partitions & q-series)**

```bash
python math_discovery_pro.py \
  --model gpt-4o \
  --domain "q-series identities and integer partitions: product-sum forms and parity constraints" \
  --rounds 1 --beam 20 --tests 400
```

Artifacts land in `results.jsonl`, `lemma_bank.jsonl`, `proofs/`, and optional
`got.graphml`. Peek at the top candidates:

```python
import json, itertools

rows = []
with open("results.jsonl", "r", encoding="utf-8") as f:
    for line in itertools.islice(f, 5000):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

rows.sort(key=lambda r: -r["score"]["total"])
for r in rows[:5]:
    print(r["conjecture"]["id"], r["status"], f"pass={r['passrate']:.2f}", r["conjecture"]["statement"])
```

### 2.2 Code construction (`code_construction_pro.py`)

**Production-minded CLI (CSV → SQLite with filters)**

```bash
python code_construction_pro.py \
  --model gpt-4o \
  --task "A CLI that imports a CSV into SQLite and supports WHERE + GROUP BY + ORDER BY via a tiny DSL" \
  --rounds 2 --outdir code_out/sqlite_cli
```

Run the generated CLI:

```bash
cd code_out/sqlite_cli
python -m cli --db data.db --import data.csv
python -m cli --db data.db --query 'FROM data WHERE col>10 GROUP BY key ORDER BY sum DESC'
```

**Minimal REST microservice (FastAPI)**

```bash
python code_construction_pro.py \
  --model gpt-4o \
  --task "A minimal FastAPI service for a todo list with tags and search; include pytest tests" \
  --rounds 2 --outdir code_out/todo_api
```

### 2.3 Theoretical physics discovery (`theoretical_physics_discovery_pro.py`)

```bash
python theoretical_physics_discovery_pro.py \
  --model gpt-5 \
  --domain "classical field theory: effective Lagrangians and dispersion relations" \
  --rounds 2
```

Signals reported per hypothesis include dimensional homogeneity, invariance
checks, and coarse limiting behaviour. Inspect a sample:

```python
import json, itertools
with open("results.jsonl", "r", encoding="utf-8") as f:
    for line in itertools.islice(f, 100):
        j = json.loads(line)
        if j.get("domain") == "theoretical_physics":
            print(j["id"], j["statement"], "dim_ok=", j["dim_ok"])
```

### 2.4 Algorithm discovery (`algorithm_discovery_pro.py`)

**Max subarray with constraints (oracle vs impl)**

```bash
python algorithm_discovery_pro.py \
  --model gpt-4o \
  --problem "maximum subarray sum where length is between L and R inclusive" \
  --rounds 2
```

**Small-graph problems (minimum vertex cover)**

```bash
python algorithm_discovery_pro.py \
  --model gpt-4o \
  --problem "minimum vertex cover on graphs up to n=12" \
  --rounds 1
```

Expect differential testing against a brute-force oracle, runtime sampling, and
automatic repair suggestions for the top candidate.

### 2.5 Unified orchestrator (`comprehensive_pro.py`)

**Let the system choose the discipline & paradigms**

```bash
python comprehensive_pro.py \
  --model gpt-4o \
  --task "derive and test congruences of central binomial coefficients mod p^2 for small primes" \
  --rounds 2 --beam 24 --parallel 8
```

**Force a domain (coding) with parallel exploration**

```bash
python comprehensive_pro.py \
  --model gpt-4o \
  --task "build a queryable time-series REST service with percentile aggregations and rolling windows" \
  --domain coding --rounds 2 --beam 16 --parallel 8
```

The orchestrator creates a structured plan, spins up multi-paradigm proposals,
logs each round to `orchestrator_logs/results.jsonl`, and highlights the best
candidate after every iteration.

---

## 3. Test-time research algorithms（研究循环）

The engines implement discipline-specific loops:

### Code Construction Pro (TDD-Refine)

1. **Planner** derives requirements, constraints, and acceptance tests.
2. **Architect** proposes minimal component/file plans.
3. **Builder** emits code + tests (structured JSON).
4. **Runner** executes tests & static checks.
5. **Critic** proposes repairs; iterate until green.

**Scoring:** pass rate, linter/type signal, complexity estimate, minimum
description length (code size).

### Theoretical Physics Discovery Pro (Symmetry-Dimensional-Limit)

1. **Planner** selects subdomain and observables.
2. **Hypothesizer** proposes relations/Lagrangians with units.
3. **Checker** enforces dimensional homogeneity, limit tests, and simple
   invariance checks (scaling/parity) via SymPy.
4. **Deriver** returns derivation plans and constraints.
5. **Critic** repairs inconsistencies; iterate.

**Scoring:** dimensional validity, invariance pass rate, limit sanity, parsimony.

### Algorithm Discovery Pro (Spec-Invariant-Oracle)

1. **Planner** crystallizes the problem specification.
2. **Explorer** proposes algorithm families with invariants & complexities.
3. **Coder** emits candidate implementations plus brute-force oracles.
4. **Tester** runs random/adversarial cases, compares outputs, and times runs.
5. **Prover** drafts proof sketches & complexity arguments.
6. **Critic** repairs failing candidates; select Pareto-best solution.

**Scoring:** correctness vs oracle, speedup, complexity claim coherence, MDL.

### Mathematics Discovery Pro (Conjecture-Falsify-Repair)

1. **Planner** identifies promising subdomains and variable regimes.
2. **Explorer** proposes SymPy-compatible conjectures.
3. **Checker** runs symbolic/numeric falsification + optional Z3.
4. **Repairer** introduces minimal extra conditions.
5. **Prover** sketches proofs and Lean skeletons; GoT captures dependencies.

**Scoring:** empirical support, novelty, simplicity, provability, efficiency.

---

## 4. Installation workflow

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Export `OPENAI_API_KEY` or place it in a `.env` file at the project root.
4. Optional tooling (auto-detected): `pytest`, `ruff`, `mypy`, `sympy`, `networkx`.

---

## 5. Artifacts and reproducibility aids

- JSONL traces capture every proposal, score, and verification signal.
- Graph-of-Thought exports (`*.graphml`) let you replay research trajectories.
- Code generators emit complete projects under `code_out/`; math proofs appear in
  `proofs/`; orchestrator sessions write to `orchestrator_logs/`.
- Seeds, beam size, rounds, and models are configurable on each CLI entrypoint.

---

## 6. Paper-ready documentation

The manuscript `comprehensive_reasoning_framework.tex` is a complete LaTeX paper
covering the framework, thinking paradigms, and case studies. Compile with
`pdflatex` (twice) or `xelatex`:

```bash
pdflatex comprehensive_reasoning_framework.tex
pdflatex comprehensive_reasoning_framework.tex
```

An optional `Makefile` snippet:

```make
PAPER=comprehensive_reasoning_framework
all:
pdflatex $(PAPER).tex
pdflatex $(PAPER).tex
clean:
rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).toc
```

---

## 7. Quick reference

- Coding automation: `code_construction_pro.py`
- Theoretical physics modeling: `theoretical_physics_discovery_pro.py`
- Algorithm synthesis: `algorithm_discovery_pro.py`
- Mathematics discovery: `math_discovery_pro.py`
- Unified orchestrator: `comprehensive_pro.py`

Each loop is engineered for **first-class research automation**: Structured
Outputs guarantee parsable artifacts, while iterative falsification keeps the
models grounded in executable feedback.
