#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comprehensive_pro.py
--------------------
Unified orchestrator that coordinates the specialized test-time reasoning
engines in this repository (math, code, physics, algorithms).  The orchestrator
combines lightweight domain classification, multi-paradigm planning, LLM-backed
artifact proposals, heuristic scoring, and JSONL logging so you can run an
end-to-end research session from a single entry-point.

Usage examples:
  python comprehensive_pro.py --model gpt-4o \
      --task "derive and test congruences of central binomial coefficients mod p^2" \
      --rounds 2 --beam 24 --parallel 8

  python comprehensive_pro.py --model gpt-4o \
      --task "build a queryable time-series REST service with percentile aggregations" \
      --domain coding --rounds 2 --beam 16 --parallel 8
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import json
import re
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - hard failure path
    raise SystemExit("Please `pip install openai>=1.40` to run comprehensive_pro.py") from exc

console = Console()

# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

PLAN_SCHEMA = {
    "name": "research_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["domain", "paradigms", "milestones", "verifiers"],
        "properties": {
            "domain": {
                "type": "string",
                "enum": ["math", "coding", "physics", "algorithms"],
            },
            "paradigms": {
                "type": "array",
                "minItems": 3,
                "maxItems": 5,
                "items": {
                    "type": "string",
                    "enum": [
                        "LinearCoT",
                        "TreeOfThought",
                        "GraphOfThought",
                        "Reflexion",
                        "MultiAgent",
                        "MetaCognitive",
                        "SelfEvolving",
                    ],
                },
            },
            "milestones": {
                "type": "array",
                "minItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "goal", "deliverables"],
                    "properties": {
                        "id": {"type": "string"},
                        "goal": {"type": "string"},
                        "deliverables": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "verifiers": {
                "type": "array",
                "minItems": 2,
                "items": {"type": "string"},
            },
        },
    },
}

PROPOSAL_SCHEMA = {
    "name": "proposal_batch",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["proposals"],
        "properties": {
            "proposals": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "id",
                        "paradigm",
                        "summary",
                        "actions",
                        "verification",
                        "signals",
                        "novelty",
                        "support",
                        "provability",
                        "efficiency",
                    ],
                    "properties": {
                        "id": {"type": "string"},
                        "paradigm": {
                            "type": "string",
                            "enum": [
                                "LinearCoT",
                                "TreeOfThought",
                                "GraphOfThought",
                                "Reflexion",
                                "MultiAgent",
                                "MetaCognitive",
                                "SelfEvolving",
                            ],
                        },
                        "summary": {"type": "string"},
                        "actions": {
                            "type": "array",
                            "minItems": 2,
                            "items": {"type": "string"},
                        },
                        "verification": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                        "signals": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                        "novelty": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "support": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "provability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "efficiency": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
            }
        },
    },
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def make_client() -> OpenAI:
    """Instantiate an OpenAI client. Assumes OPENAI_API_KEY is configured."""
    return OpenAI()


def llm_json(
    client: OpenAI,
    model: str,
    instructions: str,
    user_input: str,
    schema: Dict[str, Any],
    temperature: float = 0.4,
    max_output_tokens: int = 1800,
) -> Dict[str, Any]:
    """Call the Responses API with a JSON schema and coerce the text to JSON."""
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_format={"type": "json_schema", "json_schema": schema},
    )
    raw = response.output_text
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


@dc.dataclass
class Proposal:
    """Normalized representation of an LLM proposal."""

    id: str
    paradigm: str
    summary: str
    actions: List[str]
    verification: List[str]
    signals: List[str]
    novelty: float
    support: float
    provability: float
    efficiency: float

    def weighted_score(self) -> float:
        return (
            0.40 * self.support
            + 0.25 * self.novelty
            + 0.20 * self.provability
            + 0.15 * self.efficiency
        )


@dc.dataclass
class RoundResult:
    round_id: int
    proposals: List[Proposal]
    elapsed: float

    def topk(self, k: int = 3) -> List[Proposal]:
        return sorted(self.proposals, key=lambda p: p.weighted_score(), reverse=True)[:k]


DOMAIN_KEYWORDS = {
    "math": ["binomial", "congruence", "integer", "theorem", "proof", "algebra", "number"],
    "coding": ["api", "service", "cli", "code", "python", "rest", "app", "library"],
    "physics": ["lagrangian", "field", "dispersion", "symmetry", "hamiltonian", "particle"],
    "algorithms": ["subarray", "graph", "complexity", "dynamic", "optimization", "oracle"],
}

DEFAULT_PARADIGMS = {
    "math": ["TreeOfThought", "GraphOfThought", "Reflexion", "SelfEvolving"],
    "coding": ["MultiAgent", "Reflexion", "LinearCoT", "SelfEvolving"],
    "physics": ["LinearCoT", "MetaCognitive", "GraphOfThought", "Reflexion"],
    "algorithms": ["TreeOfThought", "LinearCoT", "Reflexion", "SelfEvolving"],
}

DEFAULT_VERIFIERS = {
    "math": ["SymPy evaluation", "numeric falsification", "optional Z3 counterexamples"],
    "coding": ["unit tests", "linters", "type checkers"],
    "physics": ["dimensional analysis", "invariance checks", "limit behavior"],
    "algorithms": ["oracle differential testing", "adversarial fuzzing", "runtime sampling"],
}


# ---------------------------------------------------------------------------
# Core orchestration logic
# ---------------------------------------------------------------------------


def classify_domain(task: str) -> str:
    task_lower = task.lower()
    scores: Dict[str, int] = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in task_lower:
                scores[domain] += 1
    best_domain = max(scores.items(), key=lambda item: item[1])[0]
    if scores[best_domain] == 0:
        return "coding"  # default bias towards code automation if unclear
    return best_domain


def build_plan(
    client: OpenAI,
    model: str,
    domain: str,
    task: str,
    rounds: int,
    beam: int,
    parallel: int,
) -> Dict[str, Any]:
    prompt = textwrap.dedent(
        f"""
        You are the orchestrator of a comprehensive multi-paradigm research system.
        Formulate a crisp plan for tackling the task below using the available
        domain engines (math, coding, physics, algorithms). Honour the beam/parallel
        constraints and produce milestones that a research engineer could execute.

        Task: {task}
        Selected domain: {domain}
        Rounds: {rounds}
        Beam width: {beam}
        Parallel workers: {parallel}
        """
    ).strip()

    plan = llm_json(
        client=client,
        model=model,
        instructions="Respond with a structured plan for the orchestrator.",
        user_input=prompt,
        schema=PLAN_SCHEMA,
    )

    return plan


def collect_proposals(
    client: OpenAI,
    model: str,
    domain: str,
    task: str,
    paradigms: Iterable[str],
    beam: int,
    round_id: int,
    previous_best: Optional[Proposal],
) -> List[Proposal]:
    context_lines = []
    if previous_best is not None:
        context_lines.append("Previous best proposal summary: \n" + previous_best.summary)
        context_lines.append(
            "Prior verification focus: \n" + "\n".join(previous_best.verification)
        )
    context = "\n\n".join(context_lines)
    prompt = textwrap.dedent(
        f"""
        Domain: {domain}
        Target task: {task}
        Active paradigms: {', '.join(paradigms)}
        Beam size (proposals requested): {beam}
        Round: {round_id}
        {context}

        Produce tightly scoped proposals that could be dispatched to the corresponding
        domain engine. Each proposal should include verification strategies and
        score estimates between 0 and 1.
        """
    ).strip()

    data = llm_json(
        client=client,
        model=model,
        instructions="Return a JSON batch of proposals.",
        user_input=prompt,
        schema=PROPOSAL_SCHEMA,
        temperature=0.6 if round_id > 1 else 0.4,
    )

    proposals = [Proposal(**payload) for payload in data["proposals"]]
    return proposals[:beam]


def log_results(path: Path, round_result: RoundResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for proposal in round_result.proposals:
            record = {
                "round": round_result.round_id,
                "elapsed": round_result.elapsed,
                "proposal": dc.asdict(proposal),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------


def render_round(result: RoundResult) -> None:
    table = Table(title=f"Round {result.round_id} proposals", show_lines=False)
    table.add_column("ID", justify="left")
    table.add_column("Paradigm")
    table.add_column("Score", justify="right")
    table.add_column("Summary", overflow="fold")
    table.add_column("Verification focus", overflow="fold")
    for proposal in sorted(result.proposals, key=lambda p: p.weighted_score(), reverse=True):
        table.add_row(
            proposal.id,
            proposal.paradigm,
            f"{proposal.weighted_score():.3f}",
            proposal.summary,
            "\n".join(proposal.verification),
        )
    console.print(table)

    best = result.topk(1)[0]
    console.print(
        Panel(
            f"Top proposal: [bold]{best.id}[/] ({best.paradigm})\n"
            f"Score={best.weighted_score():.3f}\n"
            f"Signals: {', '.join(best.signals)}",
            title="Selection",
        )
    )


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive research orchestrator")
    parser.add_argument("--task", required=True, help="High-level research goal")
    parser.add_argument("--model", default="gpt-4o", help="Model name for the Responses API")
    parser.add_argument(
        "--domain",
        choices=["math", "coding", "physics", "algorithms"],
        help="Optionally pin the domain engine",
    )
    parser.add_argument("--rounds", type=int, default=2, help="How many orchestrator rounds")
    parser.add_argument("--beam", type=int, default=8, help="Number of proposals per round")
    parser.add_argument("--parallel", type=int, default=4, help="How many candidates verify in parallel")
    parser.add_argument(
        "--outdir",
        default="orchestrator_logs",
        help="Directory for JSONL traces and supporting artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    domain = args.domain or classify_domain(args.task)
    paradigms = DEFAULT_PARADIGMS[domain][:]

    console.print(
        Panel(
            f"Task: {args.task}\n" f"Domain: {domain}\n" f"Rounds: {args.rounds} | Beam: {args.beam} | Parallel: {args.parallel}",
            title="Comprehensive Pro",
        )
    )

    client = make_client()

    plan = build_plan(
        client=client,
        model=args.model,
        domain=domain,
        task=args.task,
        rounds=args.rounds,
        beam=args.beam,
        parallel=args.parallel,
    )

    console.print(Panel(json.dumps(plan, ensure_ascii=False, indent=2), title="Plan"))

    log_path = Path(args.outdir) / "results.jsonl"
    console.print(f"Logging proposals to {log_path}")

    previous_best: Optional[Proposal] = None

    for round_id in range(1, args.rounds + 1):
        start = time.time()
        proposals = collect_proposals(
            client=client,
            model=args.model,
            domain=domain,
            task=args.task,
            paradigms=paradigms,
            beam=args.beam,
            round_id=round_id,
            previous_best=previous_best,
        )
        elapsed = time.time() - start
        result = RoundResult(round_id=round_id, proposals=proposals, elapsed=elapsed)
        log_results(log_path, result)
        render_round(result)
        previous_best = result.topk(1)[0]

    console.print(
        Panel(
            f"Best proposal across rounds: [bold]{previous_best.id}[/] ({previous_best.paradigm})\n"
            f"Score={previous_best.weighted_score():.3f}\n"
            f"Summary: {previous_best.summary}",
            title="Final Selection",
        )
    )

    console.print("Done. Inspect the JSONL log for full traces.")


if __name__ == "__main__":
    main()
