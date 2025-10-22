#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
algorithm_discovery_pro.py
--------------------------
Test-time algorithm discovery: candidates + brute-force oracle + tests + repair.
Run:
  python algorithm_discovery_pro.py --model gpt-4o --problem "minimum vertex cover on small graphs"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - guidance message only
    raise SystemExit("Please `pip install openai>=1.40`") from exc

console = Console()
client = OpenAI()

# -------- JSON Schemas --------
PLAN_SCHEMA = {
    "name": "algo_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["inputs", "outputs", "constraints", "families"],
        "properties": {
            "inputs": {"type": "string"},
            "outputs": {"type": "string"},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "families": {"type": "array", "minItems": 2, "items": {"type": "string"}},
        },
    },
}

CANDIDATES_SCHEMA = {
    "name": "algo_candidates",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "id",
                        "approach",
                        "invariant",
                        "pseudocode",
                        "python_impl",
                        "python_bruteforce",
                        "complexity",
                    ],
                    "properties": {
                        "id": {"type": "string"},
                        "approach": {"type": "string"},
                        "invariant": {"type": "string"},
                        "pseudocode": {"type": "string"},
                        "python_impl": {"type": "string"},
                        "python_bruteforce": {"type": "string"},
                        "complexity": {"type": "string"},
                    },
                },
            }
        },
    },
}

REPAIR_SCHEMA = {
    "name": "algo_repair",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["fix_impl", "rationale"],
        "properties": {
            "fix_impl": {"type": "string"},
            "rationale": {"type": "string"},
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
    "You are an algorithms researcher. Turn a natural-language problem into a precise spec "
    "(inputs, outputs, constraints) and list plausible algorithm families. JSON only."
)
EXPLORER_SYS = (
    "You propose multiple algorithm candidates. For each, provide: approach, invariant, pseudocode, "
    "Python implementation function solve(x), and a brute-force baseline function brute(x) for small inputs. JSON only."
)
CRITIC_SYS = (
    "You are an algorithms critic. Given failing cases with (input, impl_output, brute_output), "
    "produce a corrected implementation and rationale. JSON only."
)


def run_module(code: str, func_name: str, arg: Any) -> Any:
    with tempfile.TemporaryDirectory() as tmpdir:
        module_path = os.path.join(tmpdir, "mod.py")
        with open(module_path, "w", encoding="utf-8") as handle:
            handle.write(code)
        spec = importlib.util.spec_from_file_location("mod", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["mod"] = module
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        func = getattr(module, func_name)
        return func(arg)


def random_cases(problem: str, count: int = 40) -> List[Any]:
    cases: List[Any] = []
    lowered = problem.lower()
    if "array" in lowered or "subarray" in lowered:
        for _ in range(count):
            size = random.randint(1, 30)
            cases.append([random.randint(-20, 20) for _ in range(size)])
    elif "graph" in lowered:
        for _ in range(count):
            nodes = random.randint(3, 10)
            edges = []
            for i in range(nodes):
                for j in range(i + 1, nodes):
                    if random.random() < 0.25:
                        edges.append((i, j))
            cases.append({"n": nodes, "edges": edges})
    else:
        for _ in range(count):
            size = random.randint(1, 20)
            cases.append([random.randint(0, 50) for _ in range(size)])
    return cases


def evaluate_candidate(problem: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
    cases = random_cases(problem)
    successes = 0
    total = 0
    failures: List[Dict[str, Any]] = []
    start = time.time()
    for case in cases:
        total += 1
        try:
            impl_result = run_module(candidate["python_impl"], "solve", case)
            brute_result = run_module(candidate["python_bruteforce"], "brute", case)
            if impl_result == brute_result:
                successes += 1
            else:
                failures.append({"input": case, "impl": impl_result, "brute": brute_result})
        except Exception as exc:  # pragma: no cover - executes arbitrary user code
            failures.append({"input": case, "impl": f"ERROR: {exc}", "brute": "N/A"})
    duration = time.time() - start
    accuracy = successes / max(total, 1)
    return {"acc": accuracy, "n": total, "time": duration, "failures": failures[:5]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--problem", required=True, help="Natural-language description of a CS problem")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    console.rule("[bold green]Algorithm Discovery Pro[/]")

    plan = llm_json(args.model, PLANNER_SYS, f"Problem: {args.problem}", PLAN_SCHEMA, temperature=0.3)
    console.print(Panel.fit(json.dumps(plan, indent=2, ensure_ascii=False), title="Spec & Families"))

    best: Tuple[Dict[str, Any], Dict[str, Any]] | None = None
    for round_idx in range(1, args.rounds + 1):
        console.rule(f"[bold cyan]Round {round_idx}[/]")
        candidates = llm_json(
            args.model,
            EXPLORER_SYS,
            f"Problem: {args.problem}\nSpec: {json.dumps(plan, ensure_ascii=False)}",
            CANDIDATES_SCHEMA,
            temperature=0.6,
        )["candidates"]

        scored: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for candidate in candidates:
            scored.append((candidate, evaluate_candidate(args.problem, candidate)))
        scored.sort(key=lambda item: (-item[1]["acc"], item[1]["time"]))

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID")
        table.add_column("Acc")
        table.add_column("Time(s)")
        table.add_column("Fails")
        for cand, result in scored:
            table.add_row(cand["id"], f"{result['acc']:.2f}", f"{result['time']:.3f}", str(len(result["failures"])))
        console.print(table)

        top_cand, top_result = scored[0]
        if top_result["acc"] < 1.0 and top_result["failures"]:
            repair = llm_json(
                args.model,
                CRITIC_SYS,
                (
                    f"Problem: {args.problem}\n"
                    f"Candidate ID: {top_cand['id']}\n"
                    f"Implementation:\n{top_cand['python_impl']}\n"
                    f"Failures:\n{json.dumps(top_result['failures'], ensure_ascii=False)}"
                ),
                REPAIR_SCHEMA,
                temperature=0.3,
            )
            fixed = dict(top_cand)
            fixed["python_impl"] = repair["fix_impl"]
            fixed_result = evaluate_candidate(args.problem, fixed)
            console.print(Panel.fit(repair["rationale"], title="Repair Rationale"))
            console.print(Panel.fit(json.dumps(fixed_result, indent=2, ensure_ascii=False), title="After Repair"))
            if fixed_result["acc"] >= top_result["acc"]:
                scored[0] = (fixed, fixed_result)

        if best is None or scored[0][1]["acc"] > best[1]["acc"] or (
            scored[0][1]["acc"] == best[1]["acc"] and scored[0][1]["time"] < best[1]["time"]
        ):
            best = scored[0]

    console.rule("[bold green]Best Candidate[/]")
    if best:
        candidate, result = best
        console.print(Panel.fit(candidate["approach"], title=f"Approach: {candidate['id']}"))
        console.print(Panel.fit(candidate["invariant"], title="Invariant"))
        console.print(Panel.fit(candidate["pseudocode"], title="Pseudocode"))
        console.print(Panel.fit(candidate["complexity"], title="Complexity"))
        console.print(
            Panel.fit(
                f"Accuracy: {result['acc']:.2f}, Time(s): {result['time']:.3f}, Failures: {len(result['failures'])}",
                title="Evaluation",
            )
        )


if __name__ == "__main__":
    main()
