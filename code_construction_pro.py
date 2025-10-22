#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code_construction_pro.py
------------------------
Test-time code construction with planning, code emission, testing, and repair.
Run:
  python code_construction_pro.py --model gpt-4o --task "REST JSON service for todo list with search and tags"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - guidance message only
    raise SystemExit("Please `pip install openai>=1.40`") from exc

console = Console()
client = OpenAI()

# ---------- JSON Schemas (Structured Outputs) ----------
PLAN_SCHEMA = {
    "name": "plan",
    "strict": True,
    "schema": {
        "type": "object",
        "required": ["requirements", "acceptance_criteria", "components"],
        "properties": {
            "requirements": {"type": "array", "items": {"type": "string"}, "minItems": 3},
            "acceptance_criteria": {"type": "array", "items": {"type": "string"}, "minItems": 3},
            "components": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        },
        "additionalProperties": False,
    },
}

ARTIFACTS_SCHEMA = {
    "name": "artifacts",
    "strict": True,
    "schema": {
        "type": "object",
        "required": ["files", "tests", "run_cmd"],
        "properties": {
            "files": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["path", "language", "content"],
                    "properties": {
                        "path": {"type": "string"},
                        "language": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            "tests": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["path", "framework", "content"],
                    "properties": {
                        "path": {"type": "string"},
                        "framework": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            "run_cmd": {"type": "string"},
        },
        "additionalProperties": False,
    },
}

REPAIR_SCHEMA = {
    "name": "repair",
    "strict": True,
    "schema": {
        "type": "object",
        "required": ["changes", "rationale"],
        "properties": {
            "changes": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "additionalProperties": False,
    },
}

# ---------- Prompts ----------
PLANNER_SYS = (
    "You are a software architect. Given a task, produce a crisp plan with requirements, "
    "acceptance criteria, and component list (files/modules). Return JSON only."
)
BUILDER_SYS = (
    "You are a senior engineer. Produce minimal, idiomatic code and tests as JSON files. "
    "Prefer standard library; keep deps optional. Tests should be thorough and deterministic. "
    "Return JSON only."
)
CRITIC_SYS = (
    "You are a repair engineer. Given test results and failing traces, propose minimal changes "
    "and reasoning. JSON only."
)


def llm_json(
    model: str,
    instructions: str,
    user_input: str,
    schema: Dict[str, Any],
    temperature: float = 0.4,
    max_output_tokens: int = 2000,
) -> Dict[str, Any]:
    """Call the OpenAI Responses API enforcing a JSON schema."""

    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_format={"type": "json_schema", "json_schema": schema},
    )
    return json.loads(resp.output_text)


def write_artifacts(outdir: str, artifacts: Dict[str, Any]) -> None:
    """Persist generated files/tests to disk."""

    os.makedirs(outdir, exist_ok=True)
    for file_obj in artifacts["files"]:
        dst = os.path.join(outdir, file_obj["path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as handle:
            handle.write(file_obj["content"])
    for test_obj in artifacts["tests"]:
        dst = os.path.join(outdir, test_obj["path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as handle:
            handle.write(test_obj["content"])


def run_checks(outdir: str, run_cmd: str) -> Dict[str, Any]:
    """Execute generated tests and optional linting."""

    lint: Dict[str, Any] = {}
    try:
        result = subprocess.run(
            ["ruff", "check", outdir, "--select", "E,F", "--quiet"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        lint["ruff_exit"] = result.returncode
        lint["ruff_stdout"] = result.stdout[-5000:]
        lint["ruff_stderr"] = result.stderr[-3000:]
    except Exception:
        lint["ruff_exit"] = None

    try:
        if run_cmd.strip():
            proc = subprocess.run(
                run_cmd,
                shell=True,
                cwd=outdir,
                capture_output=True,
                text=True,
                timeout=180,
            )
        elif shutil.which("pytest"):
            proc = subprocess.run(
                ["pytest", "-q"],
                cwd=outdir,
                capture_output=True,
                text=True,
                timeout=180,
            )
        else:
            proc = subprocess.run(
                [sys.executable, "-m", "unittest", "discover"],
                cwd=outdir,
                capture_output=True,
                text=True,
                timeout=180,
            )
        output = proc.stdout + "\n" + proc.stderr
        passed = "passed" in output.lower() and "failed" not in output.lower()
        return {
            "passed": passed,
            "exit": proc.returncode,
            "output": output[-12000:],
            "lint": lint,
        }
    except Exception as exc:  # pragma: no cover - best effort execution
        return {"passed": False, "exit": -1, "output": str(exc), "lint": lint}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--task", required=True, help="Natural-language task description")
    parser.add_argument("--outdir", default="code_out")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    console.rule("[bold green]Code Construction Pro[/]")

    # 1) Planning
    plan = llm_json(args.model, PLANNER_SYS, f"Task: {args.task}", PLAN_SCHEMA)
    console.print(Panel.fit(json.dumps(plan, indent=2, ensure_ascii=False), title="Plan"))

    artifacts: Dict[str, Any] | None = None
    for round_idx in range(1, args.rounds + 1):
        console.rule(f"[bold cyan]Round {round_idx}[/]")

        prompt = (
            f"Task: {args.task}\n"
            f"Requirements: {plan['requirements']}\n"
            f"Acceptance criteria: {plan['acceptance_criteria']}\n"
            f"Components: {plan['components']}\n"
            "Emit artifacts with a runnable test command (run_cmd)."
        )
        artifacts = llm_json(
            args.model,
            BUILDER_SYS,
            prompt,
            ARTIFACTS_SCHEMA,
            temperature=0.5,
        )
        write_artifacts(args.outdir, artifacts)

        results = run_checks(args.outdir, artifacts.get("run_cmd", ""))
        console.print(Panel.fit(results["output"], title="Test Output", border_style="yellow"))

        if results["passed"] and results["exit"] == 0:
            console.print("[bold green]✅ All tests passed[/]")
            break

        repair = llm_json(
            args.model,
            CRITIC_SYS,
            (
                f"Task: {args.task}\n"
                f"Test results:\n{results['output'][-4000:]}\n"
                "Propose minimal changes and rationale."
            ),
            REPAIR_SCHEMA,
            temperature=0.3,
        )
        console.print(
            Panel.fit(json.dumps(repair, indent=2, ensure_ascii=False), title="Repair Plan")
        )
        plan["requirements"].append("Address: " + "; ".join(repair["changes"]))

    if artifacts:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Files")
        table.add_column("Tests")
        table.add_column("Run")
        table.add_row(
            str(len(artifacts["files"])),
            str(len(artifacts["tests"])),
            artifacts.get("run_cmd", ""),
        )
        console.print(table)


if __name__ == "__main__":
    main()
