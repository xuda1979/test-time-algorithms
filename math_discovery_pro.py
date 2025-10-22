#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
math_discovery_pro.py
---------------------
A research-grade, test-time mathematics discovery engine built around:
- LLM planning/exploration with structured JSON outputs (OpenAI Responses API)
- Symbolic/numeric falsification (SymPy) + optional SAT/SMT search (Z3)
- Minimal-condition repair loop
- Lemma bank + Graph-of-Thought DAG
- Proof planning + Lean skeleton export
Usage:
  python math_discovery_pro.py --model gpt-4o --domain "additive number theory" --rounds 2 --beam 24
Docs:
  - OpenAI Python SDK (Responses API) usage examples: https://github.com/openai/openai-python  (see README)  # noqa
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

try:
    from openai import OpenAI, AsyncOpenAI
except Exception as e:  # pragma: no cover
    raise SystemExit("Please `pip install openai` (>=1.40).")

# --- Symbolic / search backends
import sympy as sp

try:
    import z3  # optional
    HAVE_Z3 = True
except Exception:
    HAVE_Z3 = False

import networkx as nx

# ========= Utilities =========

console = Console()


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ========= Domain specs & JSON schemas for structured outputs =========

# VarSpec schema
VARSPEC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "domain", "lower", "upper"],
    "properties": {
        "name": {"type": "string", "pattern": "^[a-z][a-z0-9]*$"},
        "domain": {"type": "string", "enum": ["N", "Z"]},  # focus on integer math
        "lower": {"type": "integer"},
        "upper": {"type": "integer"},
    },
}

# Conjecture schema (SymPy‑compatible)
CONJ_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "kind", "statement", "sympy", "variables"],
    "properties": {
        "id": {"type": "string"},
        "kind": {
            "type": "string",
            "enum": [
                "identity",
                "inequality",
                "congruence",
                "divisibility",
                "recurrence",
                "bound",
                "invariant",
            ],
        },
        "statement": {"type": "string"},
        "variables": {"type": "array", "items": VARSPEC_SCHEMA, "minItems": 1, "maxItems": 4},
        "constraints": {"type": "array", "items": {"type": "string"}, "default": []},
        "sympy": {
            "type": "object",
            "additionalProperties": False,
            "required": ["relation", "lhs", "rhs"],
            "properties": {
                "relation": {
                    "type": "string",
                    "enum": ["Eq", "Le", "Lt", "Ge", "Gt", "Congruent", "Divides", "Recurrence"],
                },
                "lhs": {"type": "string"},
                "rhs": {"type": "string"},
                "mod": {"type": ["integer", "null"], "default": None},  # for Congruent
            },
        },
        "notes": {"type": "string", "default": ""},
    },
}

# Top-level explorer output schema
EXPLORER_SCHEMA = {
    "name": "conjecture_batch",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["conjectures"],
        "properties": {"conjectures": {"type": "array", "minItems": 1, "items": CONJ_SCHEMA}},
    },
    "strict": True,
}

# Planner schema
PLANNER_SCHEMA = {
    "name": "plan",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["subdomains"],
        "properties": {
            "subdomains": {
                "type": "array",
                "minItems": 3,
                "maxItems": 8,
                "items": {"type": "string"},
            }
        },
    },
    "strict": True,
}

# Proof schema
PROOF_SCHEMA = {
    "name": "proof_attempt",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "plan", "lean_skeleton"],
        "properties": {
            "status": {"type": "string", "enum": ["sketch", "gap", "refuted", "proved"]},
            "plan": {"type": "string"},
            "lean_skeleton": {"type": "string"},  # Lean 4-ish
            "micro_lemmas": {"type": "array", "items": {"type": "string"}},
        },
    },
    "strict": True,
}

# ========= OpenAI client helpers (Responses API with Structured Outputs) =========

def make_client() -> OpenAI:
    # OPENAI_API_KEY should be in env (or use dotenv)
    return OpenAI()


def llm_json(
    client: OpenAI,
    model: str,
    instructions: str,
    user_input: str,
    schema: Dict[str, Any],
    temperature: float = 0.4,
    max_output_tokens: int = 2200,
) -> Dict[str, Any]:
    """
    Call the Responses API and coerce output to JSON via json_schema.
    We tolerate minor formatting issues by re-parsing if needed.
    """
    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_format={"type": "json_schema", "json_schema": schema},
    )
    # SDK exposes `.output_text` which should be valid JSON string under structured outputs
    raw = resp.output_text
    try:
        return json.loads(raw)
    except Exception:
        # Try to extract the first JSON object via a simple bracket balance heuristic
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


# ========= Data models =========

@dc.dataclass
class VarSpec:
    name: str
    domain: str  # "N" or "Z"
    lower: int
    upper: int

    def sympy_symbol(self) -> sp.Symbol:
        if self.domain == "N":
            # Nonnegative integers; ensure bounds reflect positivity
            positive = self.lower >= 1
            return sp.Symbol(self.name, integer=True, nonnegative=not positive, positive=positive)
        return sp.Symbol(self.name, integer=True)


@dc.dataclass
class SympyRel:
    relation: str
    lhs: str
    rhs: str
    mod: Optional[int] = None


@dc.dataclass
class Conjecture:
    id: str
    kind: str
    statement: str
    variables: List[VarSpec]
    sympy: SympyRel
    constraints: List[str]
    notes: str = ""

    def key(self) -> str:
        return f"{self.kind}:{self.sympy.relation}:{self.sympy.lhs}::{self.sympy.rhs}::m={self.sympy.mod or ''}"

    def hash(self) -> str:
        return md5(self.key())


# ========= SymPy evaluation & falsification =========

ALLOWED_FUNCS = {
    # arithmetic & number theory
    "Abs": sp.Abs,
    "Mod": sp.Mod,
    "gcd": sp.gcd,
    "lcm": sp.lcm,
    "floor": sp.floor,
    "binomial": sp.binomial,
    "factorial": sp.factorial,
    "primepi": sp.primepi,
    "divisors": sp.divisors,  # for constraints only (len(divisors(n)) etc.)
    "totient": sp.totient,
    "isprime": sp.isprime,  # bool-like, use in constraints only
    "sqrt": sp.sqrt,
    "log": sp.log,
    "sin": sp.sin,
    "cos": sp.cos,
    "exp": sp.exp,
    # constants
    "pi": sp.pi,
    "E": sp.E,
}

REL_MAP = {"Eq": sp.Eq, "Le": sp.Le, "Lt": sp.Lt, "Ge": sp.Ge, "Gt": sp.Gt}


class SympyTester:
    def __init__(self, conj: Conjecture):
        self.conj = conj
        self.syms = {v.name: v.sympy_symbol() for v in conj.variables}

    def _sympify(self, s: str) -> sp.Expr:
        return sp.sympify(s, locals={**ALLOWED_FUNCS, **self.syms})

    def _relation_expr(self) -> sp.Rel:
        c = self.conj.sympy
        lhs = self._sympify(c.lhs)
        rhs = self._sympify(c.rhs)

        if c.relation in REL_MAP:
            return REL_MAP[c.relation](lhs, rhs)
        elif c.relation == "Congruent":
            if c.mod is None or int(c.mod) == 0:
                raise ValueError("Congruent relation requires nonzero 'mod'.")
            return sp.Eq(sp.Mod(lhs - rhs, c.mod), 0)
        elif c.relation == "Divides":
            # lhs | rhs  -> Mod(rhs, lhs) == 0
            return sp.Eq(sp.Mod(rhs, lhs), 0)
        elif c.relation == "Recurrence":
            # Interpret "lhs == rhs" as statement about sequences; for tests, compare numerically on grid.
            return sp.Eq(lhs, rhs)
        else:
            raise ValueError(f"Unsupported relation: {c.relation}")

    def _constraint_exprs(self) -> List[sp.Expr]:
        exprs = []
        for s in self.conj.constraints or []:
            exprs.append(sp.sympify(s, locals={**ALLOWED_FUNCS, **self.syms}))
        return exprs

    def _sample_points(self, K: int) -> Iterable[Dict[sp.Symbol, int]]:
        bounds = [(v.sympy_symbol(), v.lower, v.upper) for v in self.conj.variables]
        # Structured set: corners + small integers + a few primes/squares
        special = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        primes = [p for p in sp.primerange(2, 200)]
        squares = [i * i for i in range(1, 20)]

        def choose(lo, hi):
            pool = [x for x in special + primes + squares if lo <= x <= hi]
            if len(pool) < 5:
                pool += [random.randint(lo, hi) for _ in range(5)]
            return random.choice(pool)

        # Generate K candidate points
        for _ in range(K):
            assign = {}
            for sym, lo, hi in bounds:
                if lo > hi:
                    lo, hi = hi, lo
                assign[sym] = int(choose(lo, hi))
            yield assign

    def check(self, K: int = 256) -> Tuple[float, Optional[Dict[str, int]], List[Dict[str, int]]]:
        rel = self._relation_expr()
        constraints = self._constraint_exprs()
        ok, tried = 0, 0
        counterexamples = []
        for pt in self._sample_points(K):
            # Check constraints
            good = True
            for ce in constraints:
                try:
                    if bool(ce.subs(pt)) is False:
                        good = False
                        break
                except Exception:
                    good = False
                    break
            if not good:
                continue
            tried += 1
            try:
                holds = bool(rel.subs(pt))
            except Exception:
                holds = False
            if holds:
                ok += 1
            else:
                # return first counterexample early, but keep collecting a few
                ce_map = {str(k): int(v) for k, v in pt.items()}
                counterexamples.append(ce_map)
                if len(counterexamples) >= 3:
                    pass
        passrate = (ok / max(tried, 1))
        first_cex = counterexamples[0] if counterexamples else None
        return passrate, first_cex, counterexamples

    # Optional: limited Z3 search for counterexamples in integers
    def z3_cex(self, timeout_ms: int = 1000) -> Optional[Dict[str, int]]:
        if not HAVE_Z3:
            return None
        try:
            # Very small translator: handle +,-,*,//,%, <=,<,>=,>, =, Mod, Abs
            c = self.conj.sympy
            s = z3.Solver()
            s.set("timeout", timeout_ms)

            # Z3 Int vars with same bounds
            Z = {}
            for v in self.conj.variables:
                Z[v.name] = z3.Int(v.name)
                s.add(Z[v.name] >= v.lower)
                s.add(Z[v.name] <= v.upper)

            def sym_to_z3(expr: sp.Expr):
                if isinstance(expr, sp.Symbol):
                    return Z[expr.name]
                if isinstance(expr, sp.Integer):
                    return z3.IntVal(int(expr))
                if isinstance(expr, sp.Add):
                    return sum(sym_to_z3(a) for a in expr.args)
                if isinstance(expr, sp.Mul):
                    res = z3.IntVal(1)
                    for a in expr.args:
                        res = res * sym_to_z3(a)
                    return res
                if isinstance(expr, sp.Pow) and expr.exp.is_Integer and expr.exp >= 0:
                    base = sym_to_z3(expr.base)
                    res = z3.IntVal(1)
                    for _ in range(int(expr.exp)):
                        res = res * base
                    return res
                if expr.func == sp.Mod:
                    a, b = expr.args
                    return z3.Mod(sym_to_z3(a), sym_to_z3(b))
                if expr.func == sp.Abs:
                    a = sym_to_z3(expr.args[0])
                    return z3.If(a >= 0, a, -a)
                # fallback
                raise ValueError(f"Unsupported in z3 translator: {expr}")

            lhs = sp.sympify(c.lhs, locals=self.syms)
            rhs = sp.sympify(c.rhs, locals=self.syms)
            if c.relation == "Eq":
                target = z3.Not(sym_to_z3(lhs) == sym_to_z3(rhs))
            elif c.relation in ("Le", "Lt", "Ge", "Gt"):
                op = {"Le": z3.Not(z3.IntVal(0) <= 1), "Lt": "<", "Ge": ">=", "Gt": ">"}
                # We'll express "violate the inequality"
                if c.relation == "Le":
                    target = z3.Not(sym_to_z3(lhs) <= sym_to_z3(rhs))
                elif c.relation == "Lt":
                    target = z3.Not(sym_to_z3(lhs) < sym_to_z3(rhs))
                elif c.relation == "Ge":
                    target = z3.Not(sym_to_z3(lhs) >= sym_to_z3(rhs))
                else:
                    target = z3.Not(sym_to_z3(lhs) > sym_to_z3(rhs))
            elif c.relation == "Congruent":
                m = int(c.mod or 0)
                target = z3.Not(z3.Mod(sym_to_z3(lhs) - sym_to_z3(rhs), z3.IntVal(m)) == 0)
            else:
                return None  # skip complex cases
            # constraints as z3 Bool if simple
            for scons in self.conj.constraints or []:
                try:
                    ce = sp.sympify(scons, locals=self.syms)
                    if isinstance(ce, sp.Relational):
                        # Very limited relational -> z3
                        a, b = ce.lhs, ce.rhs
                        if ce.func == sp.Eq:
                            s.add(sym_to_z3(a) == sym_to_z3(b))
                        elif ce.func == sp.Le:
                            s.add(sym_to_z3(a) <= sym_to_z3(b))
                        elif ce.func == sp.Lt:
                            s.add(sym_to_z3(a) < sym_to_z3(b))
                        elif ce.func == sp.Ge:
                            s.add(sym_to_z3(a) >= sym_to_z3(b))
                        elif ce.func == sp.Gt:
                            s.add(sym_to_z3(a) > sym_to_z3(b))
                        else:
                            pass
                except Exception:
                    pass
            s.add(target)
            if s.check() == z3.sat:
                m = s.model()
                return {k: int(m[Z[k]].as_long()) for k in Z}
            return None
        except Exception:
            return None


# ========= Scoring, memory, graph =========

@dc.dataclass
class ScoreCard:
    passrate: float
    simplicity: float
    novelty: float
    provability: float

    def total(self, alpha=0.45, beta=0.2, gamma=0.2, delta=0.15) -> float:
        return alpha * self.passrate + beta * self.simplicity + gamma * self.novelty + delta * self.provability


class LemmaBank:
    def __init__(self, path: str = "lemma_bank.jsonl"):
        self.path = path
        ensure_dir(os.path.dirname(path) or ".")
        self._seen = set()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        self._seen.add(j.get("hash"))
                    except Exception:
                        continue

    def add(self, conj: Conjecture, status: str, meta: Dict[str, Any]) -> None:
        rec = {
            "hash": conj.hash(),
            "id": conj.id,
            "kind": conj.kind,
            "statement": conj.statement,
            "sympy": dc.asdict(conj.sympy),
            "variables": [dc.asdict(v) for v in conj.variables],
            "constraints": conj.constraints,
            "status": status,
            "meta": meta,
            "ts": time.time(),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._seen.add(conj.hash())

    def seen(self, conj: Conjecture) -> bool:
        return conj.hash() in self._seen


class GoT:
    """Graph-of-Thought store."""
    def __init__(self, path: str = "got.json"):
        self.G = nx.MultiDiGraph()
        self.path = path

    def add_conjecture(self, conj: Conjecture) -> None:
        self.G.add_node(conj.hash(), label=conj.statement, kind=conj.kind)

    def edge(self, src: Conjecture, dst: Conjecture, etype: str) -> None:
        self.G.add_edge(src.hash(), dst.hash(), type=etype)

    def save(self):
        nx.write_graphml(self.G, self.path + ".graphml")
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(nx.node_link_data(self.G), f)


# ========= LLM role prompts =========

PLANNER_SYS = (
    "You are a mathematical planner. Propose crisp subdomains/tasks inside: {domain}. "
    "Be concrete (e.g., 'congruences of binomial sums mod primes', 'bounds for divisor functions')."
    "Return only JSON."
)

EXPLORER_SYS = (
    "You are a research mathematician generating testable conjectures. "
    "Output conjectures as SymPy-compatible strings and integer var specs. "
    "Allowed functions: Abs, Mod, gcd, lcm, floor, binomial, factorial, primepi, totient. "
    "Prefer 1-3 variable statements; keep degree ≤ 4; avoid division by symbolic zero. "
    "For congruences, set relation='Congruent' and include 'mod'."
)

EMPIRICIST_TXT = (
    "You are an empiricist. Given a conjecture and any counterexamples, propose the minimal fix: "
    "strengthen hypotheses (e.g., 'n odd', 'n≥2'), restrict ranges, or adjust modulus. "
    "Return ONLY a short English rationale; code is handled by the host."
)

PROVER_SYS = (
    "You are a theorem prover. Produce a proof plan and Lean-like skeleton for the conjecture; "
    "prefer induction, generating functions, or algebraic factorization. Keep the skeleton compilable style."
)

CRITIC_TXT = (
    "You are a critic. Audit the plan for hidden assumptions (domain, monotonicity, nonzero divisors) "
    "and propose micro-lemmas to close gaps."
)

LEAN_HEADER = """-- auto-generated skeleton (Lean 4 style, non-compiling by default)
import Mathlib

/-- {name} -/
theorem {thm_name} : {prop} := by
  -- TODO: fill in proof steps suggested by LLM
  sorry
"""


# ========= Core orchestrator =========

def build_conjecture_objects(batch: Dict[str, Any]) -> List[Conjecture]:
    out = []
    for c in batch["conjectures"]:
        vars_ = [VarSpec(**v) for v in c["variables"]]
        sym = SympyRel(**c["sympy"])
        out.append(
            Conjecture(
                id=c["id"],
                kind=c["kind"],
                statement=c["statement"],
                variables=vars_,
                sympy=sym,
                constraints=c.get("constraints", []),
                notes=c.get("notes", ""),
            )
        )
    return out


def simplicity_score(conj: Conjecture) -> float:
    # MDL proxy: shorter strings + fewer variables -> closer to 1.0
    L = len(conj.sympy.lhs) + len(conj.sympy.rhs) + sum(len(v.name) for v in conj.variables)
    return max(0.0, min(1.0, 1.8 / math.log2(10 + L)))


def novelty_score(conj: Conjecture, bank: LemmaBank) -> float:
    # crude: unseen = 1.0, seen = 0.2
    return 0.2 if bank.seen(conj) else 1.0


def provability_signal(proof_json: Dict[str, Any]) -> float:
    status = proof_json.get("status", "gap")
    if status == "proved":
        return 1.0
    if status == "sketch":
        return 0.7
    if status == "gap":
        return 0.4
    return 0.1


def export_lean(conj: Conjecture, dest_dir: str = "proofs") -> str:
    ensure_dir(dest_dir)
    # make a lightweight proposition string
    var_decl = ", ".join(f"{v.name} : ℕ" if v.domain == "N" else f"{v.name} : ℤ" for v in conj.variables)
    # NOTE: This is only a sketch; mapping SymPy to Lean is nontrivial
    prop = f"-- TODO: formalize: {conj.statement}"
    thm_name = f"thm_{conj.hash()}"
    path = os.path.join(dest_dir, f"{thm_name}.lean")
    with open(path, "w", encoding="utf-8") as f:
        f.write(LEAN_HEADER.format(name=conj.statement, thm_name=thm_name, prop=prop))
    return path


def discover_loop(
    model: str,
    domain: str,
    rounds: int = 2,
    beam: int = 20,
    tests: int = 400,
    use_z3: bool = False,
    save_json: str = "results.jsonl",
):
    client = make_client()
    bank = LemmaBank("lemma_bank.jsonl")
    got = GoT("got.json")

    out_records = []

    for r in range(1, rounds + 1):
        console.rule(f"[bold green]Round {r}[/]")

        # ---- Planner ----
        plan = llm_json(
            client,
            model,
            instructions=PLANNER_SYS.format(domain=domain),
            user_input="Return a JSON list of crisp subdomains for deep traction.",
            schema=PLANNER_SCHEMA,
            temperature=0.3,
        )
        subdomains = plan["subdomains"][: min(5, len(plan["subdomains"]))]

        # ---- Explorer: for each subdomain, ask for a batch of conjectures ----
        for sub in subdomains:
            console.print(Panel.fit(f"[bold]Subdomain[/]: {sub}", border_style="cyan"))

            prompt = (
                f"Domain: {domain}\n"
                f"Focus subdomain: {sub}\n"
                f"Beam size: {beam}\n"
                "Emit conjectures as SymPy-compatible json (see schema). "
                "Examples:\n"
                "  - identity: Eq(sum(k**3, (k,1,n)), (n*(n+1)/2)**2)\n"
                "  - congruence: Eq(Mod(binomial(2*n, n), 2), 0) with relation='Congruent', mod=2\n"
                "  - divisibility: Divides: lhs | rhs  -> use relation='Divides' (lhs divides rhs)\n"
                "Variables should be integers with explicit bounds."
            )

            batch = llm_json(
                client,
                model,
                instructions=EXPLORER_SYS,
                user_input=prompt,
                schema=EXPLORER_SCHEMA,
                temperature=0.6,
            )
            conjectures = build_conjecture_objects(batch)

            # ---- Evaluate each conjecture ----
            for conj in conjectures:
                if bank.seen(conj):
                    continue

                got.add_conjecture(conj)
                tester = SympyTester(conj)
                passrate, first_cex, all_cex = tester.check(K=tests)

                z3_cex = None
                if use_z3:
                    z3_cex = tester.z3_cex(timeout_ms=1500)
                    if z3_cex is not None:
                        first_cex = first_cex or z3_cex
                        passrate = min(passrate, 0.99)  # reflect existence of cex

                # ---- Proof attempt (structured) ----
                proof = llm_json(
                    client,
                    model,
                    instructions=PROVER_SYS,
                    user_input=(
                        f"Conjecture:\n{conj.statement}\n"
                        f"SymPy relation: {conj.sympy.relation} | LHS: {conj.sympy.lhs} | RHS: {conj.sympy.rhs} | mod: {conj.sympy.mod}\n"
                        f"Variables: {', '.join([v.name for v in conj.variables])}\n"
                        f"Constraints: {conj.constraints}"
                    ),
                    schema=PROOF_SCHEMA,
                    temperature=0.2,
                )

                # ---- Critic & Repair (if counterexample exists) ----
                repair_note = ""
                if first_cex:
                    repair_note = client.responses.create(
                        model=model,
                        input=(
                            EMPIRICIST_TXT
                            + "\n\n"
                            + "Conjecture: "
                            + conj.statement
                            + "\nFirst counterexample: "
                            + json.dumps(first_cex)
                            + "\nRespond in <=3 sentences."
                        ),
                        temperature=0.4,
                        max_output_tokens=400,
                    ).output_text

                # ---- Scoring & persistence ----
                score = ScoreCard(
                    passrate=float(passrate),
                    simplicity=simplicity_score(conj),
                    novelty=novelty_score(conj, bank),
                    provability=provability_signal(proof),
                )
                total = score.total()

                status = "refuted" if first_cex else ("proved" if score.provability >= 0.95 and passrate > 0.99 else "plausible")
                bank.add(
                    conj,
                    status=status,
                    meta={
                        "round": r,
                        "subdomain": sub,
                        "passrate": passrate,
                        "first_cex": first_cex,
                        "z3_cex": z3_cex,
                        "proof": proof,
                        "repair": repair_note,
                        "score": dc.asdict(score) | {"total": total},
                    },
                )

                out_records.append(
                    {
                        "round": r,
                        "subdomain": sub,
                        "conjecture": dc.asdict(conj),
                        "status": status,
                        "passrate": passrate,
                        "first_cex": first_cex,
                        "z3_cex": z3_cex,
                        "proof": proof,
                        "repair": repair_note,
                        "score": dc.asdict(score) | {"total": total},
                    }
                )

                # Optional: export Lean skeleton for promising survivors
                if status != "refuted" and total >= 0.55:
                    lean_path = export_lean(conj)
                    console.print(f"[green]Lean skeleton:[/] {lean_path}")

                # Pretty print a compact table row
                tbl = Table(show_header=True, header_style="bold magenta")
                tbl.add_column("ID", style="cyan")
                tbl.add_column("kind")
                tbl.add_column("pass")
                tbl.add_column("simp")
                tbl.add_column("nov")
                tbl.add_column("prov")
                tbl.add_column("status")
                tbl.add_row(
                    conj.id,
                    conj.kind,
                    f"{score.passrate:.2f}",
                    f"{score.simplicity:.2f}",
                    f"{score.novelty:.2f}",
                    f"{score.provability:.2f}",
                    status,
                )
                console.print(tbl)

        got.save()

    # Save session results
    with open(save_json, "a", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    console.rule("[bold green]Done[/]")


# ========= CLI =========

def main():
    parser = argparse.ArgumentParser(description="LLM-powered math discovery at test time")
    parser.add_argument("--model", type=str, default="gpt-4o", help="e.g., gpt-4o, gpt-4o-mini, gpt-5")
    parser.add_argument("--domain", type=str, required=True, help="e.g., number theory, combinatorics")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--beam", type=int, default=16, help="conjectures per subdomain")
    parser.add_argument("--tests", type=int, default=400, help="empirical tests per conjecture")
    parser.add_argument("--use-z3", action="store_true", help="enable limited Z3 counterexample search")
    parser.add_argument("--save", type=str, default="results.jsonl")
    args = parser.parse_args()

    discover_loop(
        model=args.model,
        domain=args.domain,
        rounds=args.rounds,
        beam=args.beam,
        tests=args.tests,
        use_z3=args.use_z3,
        save_json=args.save,
    )


if __name__ == "__main__":
    main()
