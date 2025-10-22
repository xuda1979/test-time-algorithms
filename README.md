# Test-Time Mathematics Discovery Engine

This repository contains **`math_discovery_pro.py`**, a research-grade pipeline that turns state-of-the-art OpenAI models into an automated mathematics discovery assistant. It coordinates model-driven conjecture generation with symbolic verification, repair, and proof-planning support so you can explore new results rapidly and reproducibly.

## Features

- **Structured conjecture outputs** powered by the OpenAI **Responses API** with JSON Schema, producing SymPy-compatible statements ready for downstream tooling.
- **Symbolic and numeric falsification** via SymPy, with an optional Z3 back-end for SAT/SMT counterexample search.
- **Minimal-condition repair loop** that proposes hypothesis refinements when counterexamples are detected.
- **Lemma bank & Graph-of-Thought tracking** persisted as JSONL/GraphML for auditing conjecture evolution.
- **Proof planning signals** including Lean skeleton exports and plan status for rapid formalization triage.
- **CLI ergonomics** such as `--model gpt-4o`/`gpt-5`, rounds, beam width, falsification intensity, and Z3 toggles.

## Installation

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key via the `OPENAI_API_KEY` environment variable or a `.env` file in the project root.

## Usage

Run targeted discovery sessions by selecting a domain, model, and search parameters:

```bash
python math_discovery_pro.py --model gpt-4o --domain "additive number theory" --rounds 2 --beam 24 --tests 500
```

Additional flags:

- `--use-z3` – enable bounded-integer counterexample searches using the Z3 SMT solver.
- `--save` – override the default `results.jsonl` log file.

Artifacts generated during a run include:

- `lemma_bank.jsonl` – persistent conjecture ledger with status, scores, and proof notes.
- `got.json` / `got.json.graphml` – Graph-of-Thought representation of conjecture relations.
- `proofs/thm_<hash>.lean` – Lean-style proof skeletons for promising statements.

Tune the domain description for sharper conjectures (e.g., “congruences of central binomial coefficients modulo prime powers”) and increase `--tests` or enable `--use-z3` for higher validation rigor.
