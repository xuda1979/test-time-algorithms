# Test-Time Research Engines

This repository collects a family of **test-time research loops** built on OpenAI's
Responses API with **Structured Outputs**. They orchestrate planning, generation,
execution, and repair cycles across software engineering, theoretical physics,
algorithm design, and mathematics.

All tools accept `--model gpt-4o`, `--model gpt-5`, or any compatible Responses
model that supports JSON schema structured outputs. Ensure your environment
exposes `OPENAI_API_KEY` and install the dependencies in `requirements.txt`.

---

## 1. Thinking Styles by Domain（领域的“思维方式”）

### A. Coding（代码构造）

- **Spec → Design → Code → Test → Repair** loop anchored in precise contracts.
- Focus on **local reasoning** (invariants, pre/postconditions, types, effects).
- **Search operators**: template synthesis, test-driven repair, API minimization,
  and latency/complexity trade-offs.
- Heavy reliance on tool feedback: linters, type checkers, property-based tests.

### B. Theoretical Physics（理论物理）

- **Model building** through symmetries, conservation laws, and effective theories.
- **Constraint-first** reasoning with dimensional analysis, limiting behavior,
  and invariance checks (scaling, parity, gauge where appropriate).
- Emphasis on falsification: compare against known regimes, toy models, and
  consistency conditions.

### C. Algorithm Discovery（算法发现）

- **Representation + invariant design** drives data structures and loop invariants.
- Explore **competing paradigms** (greedy, DP, divide-and-conquer, randomized,
  approximation).
- Pair each candidate with a **brute-force oracle** for correctness checks on
  small instances.
- Outline **proof skeletons**: exchange arguments, potentials, amortized bounds.

---

## 2. Test-Time Research Algorithms（研究循环）

### A. Code Construction Pro (TDD-Refine)

1. **Planner** derives requirements, constraints, and acceptance tests.
2. **Architect** proposes minimal component/file plans.
3. **Builder** emits code + tests (structured JSON).
4. **Runner** executes tests & static checks.
5. **Critic** proposes repairs; iterate until green.

**Scoring:** pass rate, linter/type signal, complexity estimate, minimum
description length (code size).

### B. Theoretical Physics Discovery Pro (Symmetry-Dimensional-Limit)

1. **Planner** selects subdomain and observables.
2. **Hypothesizer** proposes relations/Lagrangians with units.
3. **Checker** enforces dimensional homogeneity, limit tests, and simple
   invariance checks (scaling/parity) via SymPy.
4. **Deriver** returns derivation plans and constraints.
5. **Critic** repairs inconsistencies; iterate.

**Scoring:** dimensional validity, invariance pass rate, limit sanity, parsimony.

### C. Algorithm Discovery Pro (Spec-Invariant-Oracle)

1. **Planner** crystallizes the problem specification.
2. **Explorer** proposes algorithm families with invariants & complexities.
3. **Coder** emits candidate implementations plus brute-force oracles.
4. **Tester** runs random/adversarial cases, compares outputs, and times runs.
5. **Prover** drafts proof sketches & complexity arguments.
6. **Critic** repairs failing candidates; select Pareto-best solution.

**Scoring:** correctness vs. oracle, speedup, complexity claim coherence, MDL.

---

## 3. Scripts

Each script is standalone, mirrors the design of `math_discovery_pro.py`, and
relies on Structured Outputs for deterministic artifact handling.

### `code_construction_pro.py`

```bash
python code_construction_pro.py --model gpt-4o --task "REST JSON service for todo list with search and tags"
```

Implements the TDD-Refine loop with optional Ruff linting and pytest/unittest
execution. Generated artifacts land in `--outdir` (default `code_out`).

### `theoretical_physics_discovery_pro.py`

```bash
python theoretical_physics_discovery_pro.py --model gpt-5 --domain "classical field theory"
```

Performs symmetry- and units-aware hypothesis exploration with SymPy-powered
checks for dimensional homogeneity, invariance, and limiting behavior, followed
by derivation planning.

### `algorithm_discovery_pro.py`

```bash
python algorithm_discovery_pro.py --model gpt-4o --problem "minimum vertex cover on small graphs"
```

Generates competing algorithms, evaluates them against brute-force oracles, and
invokes the critic to repair failing candidates.

### `math_discovery_pro.py`

```bash
python math_discovery_pro.py --model gpt-4o --domain "additive number theory" --rounds 2 --beam 24
```

A research-grade mathematics discovery engine featuring conjecture exploration,
SymPy/Z3 falsification, lemma banking, Graph-of-Thought export, and Lean proof
skeleton drafting.

---

## 4. Installation

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Export `OPENAI_API_KEY` or place it in a `.env` file at the project root.
4. Optional tooling (auto-detected): `pytest`, `ruff`, `mypy`, `sympy`, `networkx`.

---

## 5. Quick Start Summary

- Coding automation: `code_construction_pro.py`
- Theoretical physics modeling: `theoretical_physics_discovery_pro.py`
- Algorithm synthesis: `algorithm_discovery_pro.py`
- Mathematics discovery: `math_discovery_pro.py`

Each loop is engineered for **first-class research automation**: Structured
Outputs guarantee parsable artifacts, while iterative falsification keeps the
models grounded in executable feedback.
