#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theoretical_physics_discovery_pro.py
------------------------------------
Test-time discovery for theoretical physics: symmetry + dimensional analysis + limits.
Run:
  python theoretical_physics_discovery_pro.py --model gpt-5 --domain "classical field theory"
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - guidance message only
    raise SystemExit("Please `pip install openai>=1.40`") from exc

import sympy as sp
from sympy.physics import units as U

console = Console()
client = OpenAI()

# -------- JSON Schemas --------
PLAN_SCHEMA = {
    "name": "physics_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["subtopics", "observables"],
        "properties": {
            "subtopics": {"type": "array", "minItems": 3, "items": {"type": "string"}},
            "observables": {"type": "array", "minItems": 2, "items": {"type": "string"}},
        },
    },
}

VAR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "unit", "lower", "upper"],
    "properties": {
        "name": {"type": "string"},
        "unit": {"type": "string"},
        "lower": {"type": "number"},
        "upper": {"type": "number"},
    },
}

HYP_SCHEMA = {
    "name": "hypothesis_batch",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["hypotheses"],
        "properties": {
            "hypotheses": {
                "type": "array",
                "minItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "id",
                        "statement",
                        "sympy_eq",
                        "variables",
                        "constraints",
                        "expected_limits",
                        "invariances",
                    ],
                    "properties": {
                        "id": {"type": "string"},
                        "statement": {"type": "string"},
                        "sympy_eq": {"type": "string"},
                        "variables": {"type": "array", "items": VAR_SCHEMA},
                        "constraints": {"type": "array", "items": {"type": "string"}},
                        "expected_limits": {"type": "array", "items": {"type": "string"}},
                        "invariances": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        },
    },
}

DERIV_SCHEMA = {
    "name": "derivation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["plan", "assumptions", "remarks"],
        "properties": {
            "plan": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "remarks": {"type": "string"},
        },
    },
}

# -------- LLM helpers --------
def llm_json(
    model: str,
    instructions: str,
    user_input: str,
    schema: Dict[str, Any],
    temperature: float = 0.4,
    max_output_tokens: int = 2000,
) -> Dict[str, Any]:
    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_format={"type": "json_schema", "json_schema": schema},
    )
    return json.loads(resp.output_text)


PLANNER_SYS = (
    "You are a theoretical physicist. Propose concrete subtopics and key observables "
    "in the domain. JSON only."
)
HYP_SYS = (
    "You generate physically meaningful hypotheses as SymPy equations with units. "
    "Enforce dimensional homogeneity and keep variables well-bounded. JSON only."
)
DERIVER_SYS = (
    "You outline a derivation plan, listing assumptions and remarks (e.g., symmetry arguments, "
    "variational principles, limiting behaviour). JSON only."
)


UNIT_MAP = {
    "meter": U.meter,
    "metre": U.meter,
    "m": U.meter,
    "second": U.second,
    "s": U.second,
    "kilogram": U.kilogram,
    "kg": U.kilogram,
    "ampere": U.ampere,
    "a": U.ampere,
    "kelvin": U.kelvin,
    "k": U.kelvin,
    "mole": U.mole,
    "mol": U.mole,
    "candela": U.candela,
    "cd": U.candela,
    "newton": U.newton,
    "n": U.newton,
    "joule": U.joule,
    "j": U.joule,
    "watt": U.watt,
    "w": U.watt,
    "coulomb": U.coulomb,
    "c": U.coulomb,
    "volt": U.volt,
    "v": U.volt,
    "tesla": U.tesla,
    "t": U.tesla,
    "pascal": U.pascal,
    "pa": U.pascal,
}


def unit_of(name: str):
    return UNIT_MAP.get(name.lower())


def dimensional_homogeneity(eq: sp.Eq, var_units: Dict[str, Any]) -> bool:
    try:
        lhs_units = sp.dimensions(eq.lhs.subs(var_units))
        rhs_units = sp.dimensions(eq.rhs.subs(var_units))
        return lhs_units == rhs_units
    except Exception:
        return False


def invariance_checks(eq: sp.Eq, invariances: List[str], vars_spec: List[Dict[str, Any]]) -> float:
    if not vars_spec:
        return 0.0
    symbol = sp.Symbol(vars_spec[0]["name"])
    ok = 0
    total = 0
    for invariance in invariances:
        try:
            if invariance.lower().startswith("parity"):
                total += 1
                residual = sp.simplify(eq.lhs - eq.rhs)
                flipped = residual.subs({symbol: -symbol})
                ok += int(sp.simplify(residual - flipped) == 0)
            elif invariance.lower().startswith("scal"):
                total += 1
                lam = sp.symbols("λ", positive=True)
                residual = sp.simplify(eq.lhs - eq.rhs)
                scaled = sp.simplify(residual.subs({symbol: lam * symbol}))
                ok += int(sp.simplify(scaled) == residual)
        except Exception:
            continue
    return ok / max(total, 1)


def limits_ok(eq: sp.Eq, vars_spec: List[Dict[str, Any]]) -> float:
    if not vars_spec:
        return 0.0
    symbol = sp.Symbol(vars_spec[0]["name"])
    try:
        residual = sp.simplify(eq.lhs - eq.rhs)
        small = sp.simplify(residual.subs({symbol: sp.Integer(1) / sp.Integer(10**6)}))
        large = sp.simplify(residual.subs({symbol: sp.Integer(10**6)}))
        if small == 0 or large == 0:
            return 1.0
        return 0.5
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--domain", required=True, help="e.g., 'classical mechanics', 'field theory'")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()

    console.rule("[bold green]Theoretical Physics Discovery Pro[/]")

    plan = llm_json(args.model, PLANNER_SYS, f"Domain: {args.domain}", PLAN_SCHEMA, temperature=0.3)
    console.print(Panel.fit(json.dumps(plan, indent=2, ensure_ascii=False), title="Plan"))

    for round_idx in range(1, args.rounds + 1):
        console.rule(f"[bold cyan]Round {round_idx}[/]")
        hyp_batch = llm_json(
            args.model,
            HYP_SYS,
            (
                f"Domain: {args.domain}\n"
                f"Subtopics: {plan['subtopics']}\n"
                f"Observables: {plan['observables']}"
            ),
            HYP_SCHEMA,
            temperature=0.6,
        )

        for hypothesis in hyp_batch["hypotheses"]:
            console.print(Panel.fit(hypothesis["statement"], title=f"Hypothesis {hypothesis['id']}"))
            try:
                equation = sp.sympify(hypothesis["sympy_eq"], locals={"Eq": sp.Eq})
                var_units = {}
                for var in hypothesis["variables"]:
                    unit = unit_of(var["unit"])
                    if unit is not None:
                        var_units[sp.Symbol(var["name"])] = unit
                dim_ok = bool(dimensional_homogeneity(equation, var_units))
            except Exception:
                dim_ok = False
                equation = sp.Eq(sp.Integer(0), sp.Integer(0))

            inv_ok = invariance_checks(equation, hypothesis.get("invariances", []), hypothesis["variables"])
            lim_ok = limits_ok(equation, hypothesis["variables"])

            derivation = llm_json(
                args.model,
                DERIVER_SYS,
                (
                    f"Hypothesis: {hypothesis['statement']}\n"
                    f"Eq: {hypothesis['sympy_eq']}\n"
                    f"Constraints: {hypothesis['constraints']}\n"
                    f"Dim OK: {dim_ok}, Invariance score: {inv_ok:.2f}, Limit score: {lim_ok:.2f}"
                ),
                DERIV_SCHEMA,
                temperature=0.3,
            )

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Dim OK")
            table.add_column("Inv%")
            table.add_column("Limit%")
            table.add_row("✅" if dim_ok else "❌", f"{inv_ok:.2f}", f"{lim_ok:.2f}")
            console.print(table)
            console.print(Panel.fit(json.dumps(derivation, indent=2, ensure_ascii=False), title="Derivation Plan"))


if __name__ == "__main__":
    main()
