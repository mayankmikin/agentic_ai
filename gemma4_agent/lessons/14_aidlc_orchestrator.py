"""
LESSON 14 — AI-DLC: Full Adaptive Workflow (Orchestrator)
===========================================================
"AI-DLC is adaptive — stages fire only when they add value."

This lesson ties Lessons 11–13 together into a single adaptive workflow.
It introduces two new concepts not covered in the individual phase lessons:

  CONCEPT 1 — ExtensionRegistry
    AI-DLC has a core workflow and opt-in extensions (security, performance,
    accessibility, etc.).  Extensions are discovered at startup, summarised to
    the user, and loaded in full only when opted in.  This keeps the default
    workflow lean — no unnecessary rules are injected.

  CONCEPT 2 — AdaptiveStageSelector
    Not every project needs every stage.  The selector reads the intent
    classification (complexity, scope) and applies ALWAYS / CONDITIONAL logic
    to decide which stages to execute.  The execution plan is printed to the
    user before anything runs.

  CONCEPT 3 — Brownfield detection
    Greenfield: no existing code in the target namespace → run full lifecycle.
    Brownfield: existing files detected → skip Inception; focus on Construction.
    The detector scans the workspace and adjusts the plan accordingly.

Full demo scenarios:
  A) Simple request ("add a print statement") → minimal path (construction only)
  B) Full request ("build a market data API") → all stages, full adaptive run

Workflow:
  Inception → Application Design → Construction → Build-and-Test instructions

AUTO_APPROVE = True for demo mode.

Run:  python3 lessons/14_aidlc_orchestrator.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from colorama import Fore, Style, init as colorama_init
from openai import OpenAI

colorama_init(autoreset=True)

# ── Config ─────────────────────────────────────────────────────────────────────

AUTO_APPROVE = True

MODEL = "docker.io/ai/gemma4:E4B"

client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="not-needed",
)

PROJECT_ROOT      = os.path.dirname(os.path.dirname(__file__))
AIDLC_ROOT        = os.path.join(PROJECT_ROOT, "aidlc-docs")
CONSTRUCTION_ROOT = os.path.join(AIDLC_ROOT, "construction")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — model call + audit
# ══════════════════════════════════════════════════════════════════════════════

def ask_model(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=1.0,
        extra_body={"chat-template-kwargs": {"enable_thinking": True}},
    )
    content = response.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def audit_log(stage: str, event: str, detail: str = ""):
    audit_path = os.path.join(AIDLC_ROOT, "audit.md")
    os.makedirs(AIDLC_ROOT, exist_ok=True)
    ts    = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = f"## [{ts}] {stage} — {event}\n"
    if detail:
        entry += f"\n{detail.strip()}\n"
    entry += "\n---\n\n"
    with open(audit_path, "a") as f:
        f.write(entry)
    print(f"  {Fore.CYAN}[AUDIT]{Style.RESET_ALL}  {stage} → {event}")


def approval_gate(message: str) -> bool:
    print(f"\n{'─'*65}")
    print(f"  ⛔  GATE — DO NOT PROCEED until confirmed")
    print(f"  {message}")
    print(f"{'─'*65}")
    if AUTO_APPROVE:
        print(f"  {Fore.YELLOW}[AUTO-APPROVE]{Style.RESET_ALL}  Gate passed automatically.")
        return True
    return input("  Approve? (y/n): ").strip().lower() == "y"


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 1 — ExtensionRegistry
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Extension:
    """
    A single opt-in AI-DLC extension.

    The extension is described briefly (summary) so the user can decide
    whether to opt in.  The full_rules are only loaded/shown after opt-in
    to keep the default context window lean.
    """
    name:       str
    trigger:    str       # when this extension is relevant
    summary:    str       # one-line description shown before opt-in
    full_rules: list[str] # rules injected into code generation after opt-in

    def display(self, index: int):
        print(f"  [{index}] {Fore.CYAN}{self.name}{Style.RESET_ALL}")
        print(f"      Trigger:  {self.trigger}")
        print(f"      Summary:  {self.summary}")


# Built-in extension catalogue (simulates *.opt-in.md discovery)
EXTENSION_CATALOGUE: list[Extension] = [
    Extension(
        name="security",
        trigger="any project handling user data or external inputs",
        summary="Input validation, sanitised logging, wrapped error handling",
        full_rules=[
            "SEC-01: Validate all external inputs before use — reject unexpected values",
            "SEC-02: Never log raw user-supplied data; sanitise identifiers first",
            "SEC-03: All errors caught and wrapped — no raw stack traces to callers",
        ],
    ),
    Extension(
        name="performance",
        trigger="projects with latency or throughput requirements",
        summary="Lazy loading, caching strategy, async-first where beneficial",
        full_rules=[
            "PERF-01: Use generators for large data sets — avoid loading all rows into memory",
            "PERF-02: Cache repeated lookups with functools.lru_cache or a simple dict",
            "PERF-03: Profile before optimising — add timing instrumentation first",
        ],
    ),
    Extension(
        name="observability",
        trigger="projects that will run in production or be monitored",
        summary="Structured logging, execution tracing, health-check endpoint",
        full_rules=[
            "OBS-01: Log at INFO level for normal operations, ERROR for exceptions",
            "OBS-02: Include a correlation ID in every log entry",
            "OBS-03: Expose a health_check() function that returns a status dict",
        ],
    ),
]


class ExtensionRegistry:
    """
    Discovers, presents, and loads opt-in AI-DLC extensions.

    In a full AI-DLC setup, extensions are discovered from *.opt-in.md files
    in the .aidlc/ directory.  Here we use a built-in catalogue to demonstrate
    the pattern without requiring additional files.

    Key behaviour:
      - Extensions are summarised BEFORE the user opts in (lean context)
      - Full rules are only returned AFTER opt-in (deferred loading)
      - The user can opt into multiple extensions
    """

    def __init__(self):
        self._active: list[Extension] = []

    def list_available(self) -> list[Extension]:
        return EXTENSION_CATALOGUE

    def present_and_select(self) -> list[Extension]:
        """Show available extensions; let user opt in to any subset."""
        print(f"\n{'─'*65}")
        print(f"  🔌  AVAILABLE EXTENSIONS (all opt-in)")
        print(f"{'─'*65}")
        for i, ext in enumerate(EXTENSION_CATALOGUE, start=1):
            ext.display(i)
            print()

        if AUTO_APPROVE:
            # In demo mode: auto-enable security only
            selected = [e for e in EXTENSION_CATALOGUE if e.name == "security"]
            print(f"  {Fore.YELLOW}[AUTO-APPROVE]{Style.RESET_ALL}  "
                  f"Auto-enabling: {[e.name for e in selected]}")
            self._active = selected
            return selected

        print(f"  Enter extension numbers to enable (e.g. '1 3') or press Enter to skip:")
        raw = input("  → ").strip()
        if not raw:
            self._active = []
            return []

        chosen = []
        for token in raw.split():
            try:
                idx = int(token) - 1
                if 0 <= idx < len(EXTENSION_CATALOGUE):
                    chosen.append(EXTENSION_CATALOGUE[idx])
            except ValueError:
                pass

        self._active = chosen
        return chosen

    def active_rules(self) -> list[str]:
        """Return flat list of all rules from active extensions."""
        rules = []
        for ext in self._active:
            rules.extend(ext.full_rules)
        return rules

    def is_active(self, extension_name: str) -> bool:
        return any(e.name == extension_name for e in self._active)


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 2 — AdaptiveStageSelector
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StageDecision:
    stage:   str
    run:     bool
    reason:  str

    def display(self):
        icon = f"{Fore.GREEN}✅{Style.RESET_ALL}" if self.run else f"{Fore.YELLOW}⏭{Style.RESET_ALL}"
        label = "Run " if self.run else "Skip"
        print(f"  {icon}  {label}  {self.stage:<25}  {self.reason}")


STAGE_SELECTOR_SYSTEM = """You are an AI-DLC Workflow Strategist.

Given an intent classification, decide which stages to run.

AI-DLC ALWAYS/CONDITIONAL rules:
  ALWAYS run:
    - inception         (classify intent, gather requirements)
    - construction      (write code)

  CONDITIONAL:
    - application-design:
        Run  IF complexity = medium OR high
        Run  IF scope = multi_module OR full_application
        Skip IF complexity = low AND scope = single_module
    - build-and-test:
        Run  IF complexity = high
        Run  IF scope = full_application
        Skip otherwise

Output JSON (no markdown):
{
  "stages": [
    {"stage": "inception",           "run": true,  "reason": "..."},
    {"stage": "application-design",  "run": true|false, "reason": "..."},
    {"stage": "construction",        "run": true,  "reason": "..."},
    {"stage": "build-and-test",      "run": true|false, "reason": "..."}
  ]
}
"""


class AdaptiveStageSelector:
    """
    Applies AI-DLC ALWAYS/CONDITIONAL logic to decide the execution plan.

    Reads complexity and scope from the intent classification.
    Returns a list of StageDecision objects — the adaptive workflow plan.
    """

    def select(self, intent: dict) -> list[StageDecision]:
        complexity = intent.get("complexity",   "medium")
        scope      = intent.get("scope",        "full_application")
        depth      = intent.get("depth_needed", "standard")

        context = (
            f"complexity={complexity}\n"
            f"scope={scope}\n"
            f"depth_needed={depth}"
        )
        raw = ask_model(STAGE_SELECTOR_SYSTEM, context)

        # Parse JSON
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        decisions: list[StageDecision] = []

        if match:
            try:
                data = json.loads(match.group())
                for item in data.get("stages", []):
                    decisions.append(StageDecision(
                        stage  = item["stage"],
                        run    = bool(item["run"]),
                        reason = item.get("reason", ""),
                    ))
            except Exception:
                pass

        if not decisions:
            # Hard-coded fallback
            run_design = complexity in ("medium", "high") or scope != "single_module"
            run_tests  = complexity == "high" or scope == "full_application"
            decisions = [
                StageDecision("inception",          True,       "always required"),
                StageDecision("application-design", run_design, "conditional on scope/complexity"),
                StageDecision("construction",       True,       "always required"),
                StageDecision("build-and-test",     run_tests,  "conditional on scope/complexity"),
            ]

        return decisions

    def write_plan(self, decisions: list[StageDecision], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lines = [
            "# Adaptive Workflow Plan\n\n",
            f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}*\n\n",
            "---\n\n",
            "| Stage | Decision | Reason |\n",
            "|-------|----------|--------|\n",
        ]
        for d in decisions:
            icon = "✅ Run" if d.run else "⏭ Skip"
            lines.append(f"| {d.stage} | {icon} | {d.reason} |\n")
        lines.append("\n")
        with open(output_path, "w") as f:
            f.writelines(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 3 — Brownfield Detector
# ══════════════════════════════════════════════════════════════════════════════

def detect_workspace(request: str) -> dict:
    """
    Detect whether we are in a brownfield (existing code) or greenfield context.

    Brownfield indicators:
      - Python files in PROJECT_ROOT (outside lessons/ and venv/)
      - Existing portfolio.db or market_data.csv (already set up)
      - aidlc-docs/ from a previous run

    If brownfield, Inception can be abbreviated (skip clarifying questions,
    reuse existing requirements).
    """
    py_files = [
        p for p in Path(PROJECT_ROOT).glob("*.py")
        if p.name not in ("agent.py",)
    ]
    has_aidlc   = os.path.isdir(AIDLC_ROOT)
    has_db      = os.path.exists(os.path.join(PROJECT_ROOT, "portfolio.db"))
    has_csv     = os.path.exists(os.path.join(PROJECT_ROOT, "market_data.csv"))

    is_brownfield = bool(py_files) or has_aidlc

    return {
        "is_brownfield": is_brownfield,
        "has_aidlc":     has_aidlc,
        "has_db":        has_db,
        "has_csv":       has_csv,
        "py_files_found": [p.name for p in py_files],
    }


# ══════════════════════════════════════════════════════════════════════════════
# INLINE INCEPTION — lightweight version for the orchestrator
# ══════════════════════════════════════════════════════════════════════════════

INTENT_SYSTEM = """You are an AI-DLC Intent Analyzer.

Classify the user request. Output JSON only (no markdown):
{
  "request_type": "new_feature" | "bug_fix" | "refactor" | "research" | "new_project",
  "scope":        "single_module" | "multi_module" | "full_application",
  "complexity":   "low" | "medium" | "high",
  "depth_needed": "minimal" | "standard" | "comprehensive",
  "summary":      "one sentence describing what is being built"
}
"""

REQUIREMENTS_SYSTEM = """You are an AI-DLC Requirements Engineer.

Given a user request and workspace context, write a concise requirements document.

Sections: Overview, Functional Requirements (FR-01...), Non-Functional Requirements (NFR-01...),
Actors, Constraints, Out of Scope.

Output ONLY the markdown document.
"""

DESIGN_SYSTEM = """You are an AI-DLC Application Designer.

Given requirements, generate a design summary covering:
- Components (name, purpose, public methods)
- Service layer (orchestration sequence)
- Dependencies (external libraries, internal couplings)

Output a concise markdown document — this will feed Construction.
"""

CONSTRUCTION_PLAN_SYSTEM = """You are an AI-DLC Construction Planner.

Given a design summary, write a checkbox-tracked code generation plan.
Format: "- [ ] Step N: description (filename.py)"
Include: directory setup, each module, orchestrator, tests.
Maximum 8 steps.  Output ONLY the list.
"""

BUILD_TEST_SYSTEM = """You are an AI-DLC Build-and-Test Advisor.

Given the list of generated files, write build-and-test instructions covering:
1. How to run the generated code
2. What tests exist and how to run them
3. How to verify the output is correct
4. Suggested CI command

Output a markdown document: "# Build and Test Instructions"
"""


class AIDLCWorkflow:
    """
    Full AI-DLC adaptive workflow orchestrator.

    Ties together:
      - Brownfield detection
      - Extension registry (opt-in)
      - Intent classification
      - AdaptiveStageSelector (ALWAYS/CONDITIONAL logic)
      - Inline Inception → Design → Construction → Build-and-Test

    Each phase writes its artifacts to aidlc-docs/ and waits at a gate.
    Skipped stages are logged to audit.md with the reason for skipping.
    """

    def __init__(self, request: str):
        self.request    = request
        self.registry   = ExtensionRegistry()
        self.selector   = AdaptiveStageSelector()
        self.intent:    dict              = {}
        self.decisions: list[StageDecision] = []

    # ── Phase helpers ────────────────────────────────────────────────────────

    def _classify_intent(self) -> dict:
        print(f"\n  {Fore.BLUE}[INCEPTION]{Style.RESET_ALL}  Classifying intent…")
        raw   = ask_model(INTENT_SYSTEM, self.request)
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {
            "request_type": "new_project",
            "scope":        "full_application",
            "complexity":   "medium",
            "depth_needed": "standard",
            "summary":      self.request[:100],
        }

    def _run_inception(self, ws: dict) -> dict:
        """Lightweight inception: classify + generate requirements."""
        print(f"\n{'─'*65}")
        print(f"  📋  INCEPTION PHASE")
        print(f"{'─'*65}")

        self.intent = self._classify_intent()
        print(f"    complexity={self.intent['complexity']}  "
              f"scope={self.intent['scope']}  "
              f"depth={self.intent['depth_needed']}")

        # If brownfield and we already have requirements, reuse them
        req_path = os.path.join(AIDLC_ROOT, "inception", "requirements", "requirements.md")
        if ws["has_aidlc"] and os.path.exists(req_path):
            print(f"    {Fore.YELLOW}[BROWNFIELD]{Style.RESET_ALL}  "
                  f"Reusing existing requirements from previous run.")
            with open(req_path) as f:
                return {"requirements": f.read(), "reused": True}

        # Greenfield: generate fresh requirements
        context = (
            f"Request: {self.request}\n"
            f"Intent: {json.dumps(self.intent, indent=2)}\n"
            f"Workspace: {'brownfield' if ws['is_brownfield'] else 'greenfield'}\n"
            f"Has DB: {ws['has_db']}  Has CSV: {ws['has_csv']}"
        )
        requirements = ask_model(REQUIREMENTS_SYSTEM, context)

        # Save to disk
        os.makedirs(os.path.join(AIDLC_ROOT, "inception", "requirements"), exist_ok=True)
        with open(req_path, "w") as f:
            f.write("# Requirements\n\n")
            f.write(f"*{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}*\n\n---\n\n")
            f.write(requirements)

        audit_log("inception", "requirements-generated")
        return {"requirements": requirements, "reused": False}

    def _run_design(self, requirements: str) -> str:
        """Generate application design from requirements."""
        print(f"\n{'─'*65}")
        print(f"  📐  APPLICATION DESIGN PHASE")
        print(f"{'─'*65}")

        design = ask_model(DESIGN_SYSTEM, f"Requirements:\n{requirements}")

        design_path = os.path.join(AIDLC_ROOT, "design", "design-summary.md")
        os.makedirs(os.path.dirname(design_path), exist_ok=True)
        with open(design_path, "w") as f:
            f.write("# Design Summary\n\n")
            f.write(f"*{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}*\n\n---\n\n")
            f.write(design)

        audit_log("design", "design-summary-generated", f"Written to: {design_path}")
        print(f"    Written → {design_path}")
        return design

    def _run_construction(self, design_summary: str, ext_rules: list[str]) -> dict:
        """Generate checkbox plan and execute code generation."""
        print(f"\n{'─'*65}")
        print(f"  🔨  CONSTRUCTION PHASE")
        print(f"{'─'*65}")

        os.makedirs(CONSTRUCTION_ROOT, exist_ok=True)

        # Create plan
        ext_note = ""
        if ext_rules:
            ext_note = "\n\nActive extension rules:\n" + "\n".join(f"- {r}" for r in ext_rules)

        plan_text = ask_model(
            CONSTRUCTION_PLAN_SYSTEM,
            f"Design summary:\n{design_summary}{ext_note}",
        )

        # Normalise + write plan
        lines = []
        for line in plan_text.splitlines():
            line = line.strip()
            if re.match(r"-\s*\[[ x]\]\s*Step\s*\d+", line):
                lines.append(line)
        if not lines:
            lines = [
                "- [ ] Step 1: Create project directory structure",
                "- [ ] Step 2: Implement data access layer",
                "- [ ] Step 3: Implement business logic layer",
                "- [ ] Step 4: Implement presentation layer",
                "- [ ] Step 5: Implement orchestrator entry-point",
                "- [ ] Step 6: Write unit tests",
            ]

        plan_path = os.path.join(CONSTRUCTION_ROOT, "code-generation-plan.md")
        with open(plan_path, "w") as f:
            f.write("# Code Generation Plan\n\n")
            f.write(f"*{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}*\n\n---\n\n")
            f.write("\n".join(lines) + "\n")

        print(f"\n  Generated {len(lines)}-step plan:")
        for line in lines:
            print(f"    {line}")

        approved = approval_gate(
            "Review code-generation-plan.md — approve to start code generation."
        )
        if not approved:
            audit_log("construction", "plan-rejected")
            return {"approved": False}

        # Execute each step
        code_root = os.path.join(PROJECT_ROOT, "market_api")
        os.makedirs(code_root, exist_ok=True)
        generated_files = []

        FILE_MAP = {
            "data":           "data_reader.py",
            "business":       "business_logic.py",
            "presentation":   "presenter.py",
            "orchestrat":     "basic_agent_with_tool_call.py",
            "entry":          "basic_agent_with_tool_call.py",
            "test":           "test_suite.py",
        }

        for step_line in lines:
            step_match = re.search(r"Step\s*(\d+)", step_line, re.IGNORECASE)
            if not step_match:
                continue
            step_n = int(step_match.group(1))
            desc   = step_line[step_line.find(":")+1:].strip()
            print(f"\n  ── Executing Step {step_n}: {desc[:60]}")

            if "directory" in desc.lower() or step_n == 1:
                init_p = os.path.join(code_root, "__init__.py")
                if not os.path.exists(init_p):
                    with open(init_p, "w") as f:
                        f.write('"""AI-DLC generated package."""\n')
                    generated_files.append(init_p)
                print(f"    Created directory → {code_root}/")
            else:
                # Determine output filename
                target = None
                for keyword, fname in FILE_MAP.items():
                    if keyword in desc.lower():
                        target = fname
                        break
                if not target:
                    target = f"module_step{step_n}.py"

                out_path = os.path.join(code_root, target)
                ext_instruction = (
                    f"\n\nApply these extension rules:\n" +
                    "\n".join(f"- {r}" for r in ext_rules)
                ) if ext_rules else ""

                code = ask_model(
                    "You are a Python code generator. Generate clean, working Python 3 code. "
                    "Output ONLY the source code, no markdown fences.",
                    f"Design summary:\n{design_summary}\n\n"
                    f"Generate code for this step:\n{desc}{ext_instruction}\n\n"
                    f"Workspace: {PROJECT_ROOT}",
                )
                code = re.sub(r"^```python\n?", "", code, flags=re.MULTILINE)
                code = re.sub(r"^```\n?",       "", code, flags=re.MULTILINE)

                with open(out_path, "w") as f:
                    f.write(code)
                generated_files.append(out_path)
                print(f"    Written → {out_path}")
                audit_log("construction", f"file-generated: {target}")

            # Mark checkbox immediately
            with open(plan_path) as pf:
                plan_content = pf.read()
            plan_content = re.sub(
                rf"(-\s*\[)\s*(\]\s*Step\s*{step_n}[:\s])",
                r"\1x\2",
                plan_content,
            )
            with open(plan_path, "w") as pf:
                pf.write(plan_content)
            print(f"  {Fore.GREEN}[✓]{Style.RESET_ALL}  Step {step_n} marked complete")

        return {"approved": True, "code_root": code_root, "files": generated_files}

    def _run_build_and_test(self, code_root: str, files: list) -> str:
        """Generate build-and-test instructions."""
        print(f"\n{'─'*65}")
        print(f"  🧪  BUILD AND TEST PHASE")
        print(f"{'─'*65}")

        file_list = "\n".join(os.path.basename(f) for f in files)
        instructions = ask_model(
            BUILD_TEST_SYSTEM,
            f"Generated files:\n{file_list}\n\n"
            f"Code directory: {code_root}\n"
            f"Project root: {PROJECT_ROOT}",
        )

        out_path = os.path.join(AIDLC_ROOT, "build-and-test-instructions.md")
        with open(out_path, "w") as f:
            f.write(instructions)

        audit_log("build-and-test", "instructions-generated", f"Written to: {out_path}")
        print(f"    Written → {out_path}")
        return out_path

    # ── Main orchestrator ────────────────────────────────────────────────────

    def run(self):
        print(f"\n{'═'*65}")
        print(f"  AI-DLC FULL ADAPTIVE WORKFLOW")
        print(f"{'═'*65}")
        print(f"  Request: {self.request}")
        print(f"{'─'*65}")

        os.makedirs(AIDLC_ROOT, exist_ok=True)
        audit_log("orchestrator", "workflow-started", f"Request: {self.request}")

        # ── Detect workspace ───────────────────────────────────────────────
        ws = detect_workspace(self.request)
        context_label = "BROWNFIELD" if ws["is_brownfield"] else "GREENFIELD"
        print(f"\n  Workspace: {Fore.CYAN}{context_label}{Style.RESET_ALL}")
        if ws["py_files_found"]:
            print(f"    Existing files: {ws['py_files_found']}")

        # ── Extensions ────────────────────────────────────────────────────
        active_extensions = self.registry.present_and_select()
        ext_rules         = self.registry.active_rules()

        if active_extensions:
            print(f"\n  Active extensions: {[e.name for e in active_extensions]}")
            print(f"  Rules to apply ({len(ext_rules)}):")
            for rule in ext_rules:
                print(f"    • {rule}")

        # ── Classify intent ───────────────────────────────────────────────
        self.intent = self._classify_intent()
        print(f"\n  Intent: {json.dumps(self.intent, indent=4)}")
        audit_log("orchestrator", "intent-classified", json.dumps(self.intent))

        # ── Select stages ─────────────────────────────────────────────────
        self.decisions = self.selector.select(self.intent)

        print(f"\n  Adaptive execution plan:")
        for d in self.decisions:
            d.display()

        # Write plan to disk
        plan_path = os.path.join(AIDLC_ROOT, "adaptive-workflow-plan.md")
        self.selector.write_plan(self.decisions, plan_path)
        audit_log("orchestrator", "adaptive-plan-written", f"Written to: {plan_path}")

        # Gate: review the adaptive plan before proceeding
        approved = approval_gate(
            "Review the adaptive workflow plan above — approve to begin execution."
        )
        if not approved:
            print(f"\n  {Fore.RED}Workflow cancelled.{Style.RESET_ALL}")
            audit_log("orchestrator", "workflow-cancelled")
            return

        # ── Execute stages ─────────────────────────────────────────────────
        stage_map = {d.stage: d for d in self.decisions}
        results   = {}

        # Stage: Inception
        if stage_map.get("inception", StageDecision("inception", True, "")).run:
            inception_result  = self._run_inception(ws)
            results["requirements"] = inception_result["requirements"]
        else:
            audit_log("inception", "phase-skipped", "Not required for this request.")
            print(f"\n  ⏭  Inception skipped")
            results["requirements"] = ""

        # Stage: Application Design
        if stage_map.get("application-design", StageDecision("application-design", False, "")).run:
            design = self._run_design(results.get("requirements", ""))
            results["design"] = design
            approved = approval_gate("Review design-summary.md — approve to proceed.")
            if not approved:
                print(f"  {Fore.RED}Design rejected.{Style.RESET_ALL}")
                audit_log("design", "phase-rejected")
                return
        else:
            audit_log("design", "phase-skipped", "Not required for this complexity/scope.")
            print(f"\n  ⏭  Application Design skipped")
            results["design"] = results.get("requirements", "")

        # Stage: Construction
        if stage_map.get("construction", StageDecision("construction", True, "")).run:
            construction_result = self._run_construction(
                results.get("design", ""),
                ext_rules,
            )
            if not construction_result.get("approved"):
                print(f"  {Fore.RED}Construction rejected.{Style.RESET_ALL}")
                return
            results["code_root"] = construction_result.get("code_root", "")
            results["files"]     = construction_result.get("files", [])
        else:
            audit_log("construction", "phase-skipped", "Not required.")
            print(f"\n  ⏭  Construction skipped")

        # Stage: Build and Test
        if stage_map.get("build-and-test", StageDecision("build-and-test", False, "")).run:
            instructions_path = self._run_build_and_test(
                results.get("code_root", PROJECT_ROOT),
                results.get("files", []),
            )
            results["build_instructions"] = instructions_path
        else:
            audit_log("build-and-test", "phase-skipped", "Not required for this scope.")
            print(f"\n  ⏭  Build-and-Test skipped")

        # ── Summary ────────────────────────────────────────────────────────
        audit_log("orchestrator", "workflow-complete")

        print(f"\n{'═'*65}")
        print(f"  ✅  AI-DLC WORKFLOW COMPLETE")
        print(f"{'═'*65}")
        print(f"  Artifacts in: {AIDLC_ROOT}/")
        print(f"    audit.md")
        print(f"    adaptive-workflow-plan.md")

        stages_run = [d.stage for d in self.decisions if d.run]
        for stage in stages_run:
            print(f"    {stage}/")

        if results.get("code_root") and os.path.exists(results["code_root"]):
            gen_files = list(Path(results["code_root"]).glob("*.py"))
            print(f"\n  Generated code in: {results['code_root']}/")
            for gf in sorted(gen_files):
                print(f"    {gf.name}")

        print(f"\n  Stages executed:  {stages_run}")
        skipped = [d.stage for d in self.decisions if not d.run]
        if skipped:
            print(f"  Stages skipped:   {skipped}  (adaptive — not needed)")
        print(f"{'═'*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO RUNNER — two scenarios
# ══════════════════════════════════════════════════════════════════════════════

def demo_full_request():
    """
    Scenario B: Full complex request → all stages run.
    """
    workflow = AIDLCWorkflow(
        "Using AI-DLC, build a market data API for our existing portfolio system "
        "that fetches live prices, caches results for 60 seconds, and exposes a "
        "REST endpoint — it needs to handle concurrent requests safely."
    )
    workflow.run()


def demo_simple_request():
    """
    Scenario A: Simple request → adaptive (fewer stages).
    """
    workflow = AIDLCWorkflow(
        "Add a function to the existing portfolio calculator that formats a "
        "holding's P&L as a colour-coded string."
    )
    workflow.run()


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        print("Running Scenario A — simple request (adaptive / fewer stages)")
        demo_simple_request()
    else:
        print("Running Scenario B — full complex request (all stages)")
        print("Pass 'simple' as argument for Scenario A.")
        demo_full_request()
