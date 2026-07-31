"""
LESSON 13 — AI-DLC: Construction (Plan-Then-Generate)
=======================================================
"Never generate code without a plan. Two-part pattern: Plan → Approve → Execute."

This lesson implements the Construction phase of AI-DLC, which has a strict rule:

  NEVER generate production code without a checkbox-tracked plan that has been
  reviewed first.  Code is generated step-by-step, one checkbox at a time.
  Each checkbox is marked [x] immediately after the step completes.

This prevents the most common LLM coding failure mode:
  jumping straight to code → generating the wrong structure → expensive rework.

Three sub-patterns implemented:
  PATTERN 1 — CodeGenerationPlanner
    Gemma4 writes a numbered, checkbox-tracked plan file.
    Each line is:  "- [ ] Step N: description"
    The plan is written to aidlc-docs/construction/code-generation-plan.md
    and MUST be approved before any code is generated.

  PATTERN 2 — CheckboxTracker
    After each step completes, a Python utility rewrites the plan file to
    mark the step [x].  This is mandatory — it keeps the plan and reality
    in sync, and gives a recoverable trail if the process is interrupted.

  PATTERN 3 — NFRAdvisor + Security Extension
    Before generating code, Gemma4 assesses NFRs (security, performance).
    If security concerns are found, the user is offered the SECURITY extension.
    When opted-in, three extra rules are enforced:
      SEC-01: Validate all external inputs before use
      SEC-02: Never log raw user data; sanitise before logging
      SEC-03: All errors caught and wrapped (no stack traces to user)

Artifacts produced (all inside aidlc-docs/):
  construction/code-generation-plan.md    ← step-by-step plan with checkboxes
  construction/nfr-assessment.md          ← security / performance analysis

Generated code lands in: gemma4_agent/portfolio_calculator/
  __init__.py
  data_reader.py         (PortfolioReader + MarketDataReader)
  calculator.py          (PnLCalculator)
  printer.py             (SummaryPrinter)
  run_summary.py         (orchestrator entry-point)
  test_calculator.py     (unit tests)

AUTO_APPROVE = True for demo mode.

Run:  python3 lessons/13_aidlc_construction.py
"""

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

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

PROJECT_ROOT       = os.path.dirname(os.path.dirname(__file__))
AIDLC_ROOT         = os.path.join(PROJECT_ROOT, "aidlc-docs")
CONSTRUCTION_ROOT  = os.path.join(AIDLC_ROOT, "construction")
CODE_OUTPUT_ROOT   = os.path.join(PROJECT_ROOT, "portfolio_calculator")

# ── Sample design documents (used when Lesson 12 output is absent) ────────────

SAMPLE_DESIGN_SUMMARY = """
Components:
- PortfolioReader: loads holdings (ticker, shares, avg_cost) from portfolio.db
- MarketDataReader: reads current prices from market_data.csv
- PnLCalculator: computes daily P&L and enriches holdings list
- SummaryPrinter: renders formatted terminal output with colorama

Services:
- run_summary(): orchestrates Reader → Reader → Calculator → Printer

Dependencies:
- External: csv (stdlib), sqlite3 (stdlib), colorama
- Internal: run_summary → all components; PnLCalculator depends on both readers' output

DB path: portfolio.db  CSV path: market_data.csv
""".strip()


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


def load_design_summary() -> str:
    """Load design docs from Lesson 12 or fall back to sample."""
    comp_path = os.path.join(AIDLC_ROOT, "design", "components.md")
    svc_path  = os.path.join(AIDLC_ROOT, "design", "services.md")
    dep_path  = os.path.join(AIDLC_ROOT, "design", "component-dependency.md")
    if all(os.path.exists(p) for p in [comp_path, svc_path, dep_path]):
        parts = []
        for p in [comp_path, svc_path, dep_path]:
            with open(p) as f:
                parts.append(f.read())
        return "\n\n".join(parts)
    print(f"  {Fore.YELLOW}[INFO]{Style.RESET_ALL}  "
          f"No Lesson 12 design docs found — using built-in sample.")
    return SAMPLE_DESIGN_SUMMARY


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 1 — CodeGenerationPlanner
# ══════════════════════════════════════════════════════════════════════════════

PLANNER_SYSTEM = """You are an AI-DLC Construction Planner.

Given a design summary, write a checkbox-tracked code generation plan.

Rules:
- Each step begins with: - [ ] Step N: description
- Include EXACTLY these step types (adapt descriptions to the project):
    Step 1: Create project directory structure
    Step 2: Implement data access components
    Step 3: Implement business logic components
    Step 4: Implement presentation/output components
    Step 5: Implement orchestrator / entry-point
    Step 6: Write unit tests
- Maximum 8 steps total
- Each description must be specific (include file names, component names)
- Output ONLY the checkbox list — no prose
"""


class CodeGenerationPlanner:
    """
    Pattern 1: Generate a checkbox-tracked construction plan.

    AI-DLC principle: "Never generate code without a plan."
    The plan is written to disk BEFORE any code is generated.
    It must be approved at the gate — no exceptions.
    """

    def __init__(self, docs_root: str):
        self.plan_path = os.path.join(docs_root, "code-generation-plan.md")

    def create_plan(self, design_summary: str) -> str:
        print(f"\n{Fore.BLUE}▶ CodeGenerationPlanner{Style.RESET_ALL}  creating construction plan…")
        os.makedirs(os.path.dirname(self.plan_path), exist_ok=True)

        raw_plan = ask_model(PLANNER_SYSTEM, f"Design summary:\n{design_summary}")

        # Normalise to ensure each step is "- [ ] Step N:"
        lines = []
        for line in raw_plan.splitlines():
            line = line.strip()
            if not line:
                continue
            # Ensure it looks like a checkbox step
            if re.match(r"-\s*\[[ x]\]\s*Step\s*\d+", line, re.IGNORECASE):
                lines.append(line)
            elif re.match(r"\d+\.", line):
                # Convert "1. ..." to "- [ ] Step 1: ..."
                step_text = re.sub(r"^\d+\.\s*", "", line)
                n         = len(lines) + 1
                lines.append(f"- [ ] Step {n}: {step_text}")

        if not lines:
            # Fallback if model returned something unexpected
            lines = [
                "- [ ] Step 1: Create portfolio_calculator/ package directory",
                "- [ ] Step 2: Implement data_reader.py (PortfolioReader, MarketDataReader)",
                "- [ ] Step 3: Implement calculator.py (PnLCalculator)",
                "- [ ] Step 4: Implement printer.py (SummaryPrinter)",
                "- [ ] Step 5: Implement run_summary.py (orchestrator)",
                "- [ ] Step 6: Write test_calculator.py (unit tests)",
            ]

        header = (
            "# Code Generation Plan\n\n"
            "*AI-DLC Construction — Plan MUST be approved before code generation begins.*\n\n"
            f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}*\n\n"
            "---\n\n"
        )
        with open(self.plan_path, "w") as f:
            f.write(header + "\n".join(lines) + "\n")

        audit_log("construction", "plan-created", f"Written to: {self.plan_path}")
        print(f"    Written → {self.plan_path}")
        for line in lines:
            print(f"    {line}")
        return self.plan_path

    def get_steps(self) -> list[str]:
        """Return list of step descriptions (unchecked steps only)."""
        with open(self.plan_path) as f:
            content = f.read()
        steps = []
        for line in content.splitlines():
            if re.match(r"-\s*\[\s*\]\s*Step\s*\d+", line):
                steps.append(line.strip())
        return steps


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 2 — CheckboxTracker
# ══════════════════════════════════════════════════════════════════════════════

class CheckboxTracker:
    """
    Pattern 2: Mark a plan step [x] immediately after it completes.

    AI-DLC mandate: "Update checkboxes IMMEDIATELY after each step — not at the end."

    This creates a recoverable execution trail.  If the process crashes mid-way,
    re-running will skip already-completed steps (the [x] markers show what's done).
    """

    def __init__(self, plan_path: str):
        self.plan_path = plan_path

    def mark_done(self, step_number: int):
        """Mark Step N as complete in the plan file."""
        with open(self.plan_path) as f:
            content = f.read()

        # Replace "- [ ] Step N:" with "- [x] Step N:"
        updated = re.sub(
            rf"(-\s*\[)\s*(\]\s*Step\s*{step_number}[:\s])",
            r"\1x\2",
            content,
        )
        with open(self.plan_path, "w") as f:
            f.write(updated)

        print(f"  {Fore.GREEN}[✓]{Style.RESET_ALL}  Step {step_number} marked complete in plan")

    def get_completed_steps(self) -> list[int]:
        """Return list of step numbers already marked [x]."""
        with open(self.plan_path) as f:
            content = f.read()
        completed = []
        for match in re.finditer(r"-\s*\[x\]\s*Step\s*(\d+)", content, re.IGNORECASE):
            completed.append(int(match.group(1)))
        return completed


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 3 — NFRAdvisor + Security Extension
# ══════════════════════════════════════════════════════════════════════════════

NFR_SYSTEM = """You are an AI-DLC NFR (Non-Functional Requirements) Advisor.

Given a design summary, assess the NFRs in two categories:

## Performance
(rate: low / medium / high risk; give one action if medium/high)

## Security
(rate: low / medium / high risk; list 1–3 concrete concerns if medium/high)

## Security Extension Recommended?
Answer: YES or NO
Reason: one sentence

Output ONLY this structured assessment — no other text.
"""

# Security rules that are enforced when the extension is opted in
SECURITY_RULES = {
    "SEC-01": "Validate all external inputs before use — reject/sanitise unexpected values",
    "SEC-02": "Never log raw user data; sanitise or hash identifiers before logging",
    "SEC-03": "All errors caught with try/except; no raw stack traces surfaced to callers",
}


class NFRAdvisor:
    """
    Pattern 3: Assess NFRs and offer the Security extension.

    AI-DLC extensions are opt-in.  The advisor runs before code generation
    and asks the user whether to apply the security rule set.

    When the security extension is active, CodeGenerationExecutor will
    apply all three SEC-* rules to every generated file.
    """

    def __init__(self, docs_root: str):
        self.nfr_path = os.path.join(docs_root, "nfr-assessment.md")

    def assess(self, design_summary: str) -> dict:
        print(f"\n{Fore.BLUE}▶ NFRAdvisor{Style.RESET_ALL}  assessing NFRs…")
        os.makedirs(os.path.dirname(self.nfr_path), exist_ok=True)

        assessment = ask_model(NFR_SYSTEM, f"Design summary:\n{design_summary}")

        with open(self.nfr_path, "w") as f:
            f.write("# NFR Assessment\n\n")
            f.write(f"*{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}*\n\n")
            f.write("---\n\n")
            f.write(assessment)
            f.write("\n")

        # Detect security recommendation
        security_recommended = bool(
            re.search(r"Security Extension Recommended\?.*YES", assessment, re.IGNORECASE | re.DOTALL)
        )

        audit_log("construction", "nfr-assessed",
                  f"security_recommended={security_recommended}")
        print(f"    Written → {self.nfr_path}")
        print(f"    Security extension recommended: {security_recommended}")
        return {"security_recommended": security_recommended, "assessment": assessment}

    def opt_in_security(self, nfr_result: dict) -> bool:
        """Ask user whether to enable the security extension."""
        if not nfr_result["security_recommended"]:
            print(f"    {Fore.GREEN}No security concerns — extension not needed{Style.RESET_ALL}")
            return False

        print(f"\n  {'─'*60}")
        print(f"  🔒  SECURITY EXTENSION AVAILABLE")
        print(f"  The NFR assessment recommends the security extension.")
        print(f"  Rules that will be applied:")
        for rule, desc in SECURITY_RULES.items():
            print(f"    {rule}: {desc}")
        print(f"  {'─'*60}")

        if AUTO_APPROVE:
            print(f"  {Fore.YELLOW}[AUTO-APPROVE]{Style.RESET_ALL}  Security extension ENABLED")
            return True
        return input("  Enable security extension? (y/n): ").strip().lower() == "y"


# ══════════════════════════════════════════════════════════════════════════════
# CODE GENERATION — per-file generators
# ══════════════════════════════════════════════════════════════════════════════

CODE_GEN_SYSTEM = """You are an AI-DLC Code Generator.

Generate clean, working Python code for the described component.

Rules:
- Python 3.10+
- Include all imports at the top
- Add a module-level docstring (one sentence)
- No placeholder comments like "TODO" or "implement this"
- If security extension is active, apply these rules to every function:
  SEC-01: Validate all external inputs before use
  SEC-02: Never log raw user data; sanitise identifiers before logging
  SEC-03: Wrap all operations in try/except; never surface raw tracebacks
- Output ONLY the Python source code (no markdown fences)
"""


def generate_code_file(component_description: str, security: bool) -> str:
    """Ask Gemma4 to generate Python code for a single component."""
    sec_note = (
        "\n\nSecurity extension is ACTIVE — enforce SEC-01, SEC-02, SEC-03 throughout."
        if security else ""
    )
    return ask_model(
        CODE_GEN_SYSTEM,
        f"{component_description}{sec_note}",
    )


class CodeGenerationExecutor:
    """
    Reads the plan step by step and generates actual Python code to CODE_OUTPUT_ROOT.

    For each step:
      1. Determine what to generate (from step description)
      2. Call Gemma4 to generate the code
      3. Write the file
      4. Immediately mark the step [x] in the plan

    AI-DLC rule: steps fire in order; no step is skipped; checkboxes are updated immediately.
    """

    def __init__(
        self,
        planner: CodeGenerationPlanner,
        tracker: CheckboxTracker,
        design_summary: str,
        security_enabled: bool,
    ):
        self.planner          = planner
        self.tracker          = tracker
        self.design_summary   = design_summary
        self.security_enabled = security_enabled

    def _step_context(self, step_line: str) -> str:
        """Build generation context from a step line and the overall design."""
        return (
            f"Design context:\n{self.design_summary}\n\n"
            f"Generate the code for this specific step:\n{step_line}\n\n"
            f"Target directory: portfolio_calculator/\n"
            f"Project root: {PROJECT_ROOT}\n"
            f"CSV path: {os.path.join(PROJECT_ROOT, 'market_data.csv')}\n"
            f"DB path: {os.path.join(PROJECT_ROOT, 'portfolio.db')}\n"
        )

    def _target_file(self, step_line: str) -> str | None:
        """
        Infer the output filename from the step description.
        Returns None for steps that don't produce a single file (e.g. "create directory").
        """
        line_lower = step_line.lower()
        if "directory" in line_lower or "structure" in line_lower:
            return None  # handled separately
        if "data_reader" in line_lower or "portfolioreader" in line_lower.replace(" ", ""):
            return "data_reader.py"
        if "calculator" in line_lower and "test" not in line_lower:
            return "calculator.py"
        if "printer" in line_lower or "summary" in line_lower and "run" not in line_lower:
            return "printer.py"
        if "run_summary" in line_lower or "orchestrat" in line_lower or "entry" in line_lower:
            return "run_summary.py"
        if "test" in line_lower:
            return "test_calculator.py"
        return None

    def execute(self):
        print(f"\n{Fore.BLUE}▶ CodeGenerationExecutor{Style.RESET_ALL}  executing plan…")

        already_done = self.tracker.get_completed_steps()
        if already_done:
            print(f"  Resuming — steps already done: {already_done}")

        steps = self.planner.get_steps()  # only unchecked steps

        for step_line in steps:
            match = re.search(r"Step\s*(\d+)", step_line, re.IGNORECASE)
            if not match:
                continue
            step_number = int(match.group(1))
            print(f"\n  ── Step {step_number}: {step_line[step_line.find(':')+1:].strip()}")

            # Step 1 = create directory structure (no code generation)
            if step_number == 1 or "directory" in step_line.lower():
                os.makedirs(CODE_OUTPUT_ROOT, exist_ok=True)
                init_path = os.path.join(CODE_OUTPUT_ROOT, "__init__.py")
                if not os.path.exists(init_path):
                    with open(init_path, "w") as f:
                        f.write('"""portfolio_calculator — AI-DLC generated package."""\n')
                print(f"    Created → {CODE_OUTPUT_ROOT}/")
                self.tracker.mark_done(step_number)
                continue

            # All other steps = generate code
            target = self._target_file(step_line)
            if not target:
                print(f"    {Fore.YELLOW}[SKIP]{Style.RESET_ALL}  Cannot infer target file for this step")
                self.tracker.mark_done(step_number)
                continue

            out_path = os.path.join(CODE_OUTPUT_ROOT, target)
            print(f"    Generating → {out_path}")

            context  = self._step_context(step_line)
            code     = generate_code_file(context, self.security_enabled)

            # Strip markdown fences if model added them
            code = re.sub(r"^```python\n?", "", code, flags=re.MULTILINE)
            code = re.sub(r"^```\n?",       "", code, flags=re.MULTILINE)

            with open(out_path, "w") as f:
                f.write(code)

            audit_log("construction", f"file-generated: {target}",
                      f"security_enabled={self.security_enabled}")

            # Mark checkbox IMMEDIATELY after file is written
            self.tracker.mark_done(step_number)

        print(f"\n  {Fore.GREEN}All plan steps executed.{Style.RESET_ALL}")


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — run_construction_phase
# ══════════════════════════════════════════════════════════════════════════════

def run_construction_phase(design_summary: str | None = None):
    """
    Run the full AI-DLC Construction phase.

    Steps:
      1. Load design documents
      2. Assess NFRs; offer security extension
      3. Create checkbox-tracked code generation plan
      4. Gate: approve plan before any code is generated
      5. Execute plan step by step with immediate checkbox updates
    """
    print(f"\n{'═'*65}")
    print(f"  AI-DLC CONSTRUCTION PHASE")
    print(f"{'═'*65}")

    if design_summary is None:
        design_summary = load_design_summary()

    os.makedirs(CONSTRUCTION_ROOT, exist_ok=True)
    audit_log("construction", "phase-started")

    # ── Step 1: Assess NFRs ────────────────────────────────────────────────
    nfr_advisor      = NFRAdvisor(CONSTRUCTION_ROOT)
    nfr_result       = nfr_advisor.assess(design_summary)
    security_enabled = nfr_advisor.opt_in_security(nfr_result)

    if security_enabled:
        print(f"\n  🔒  Security rules active:")
        for rule, desc in SECURITY_RULES.items():
            print(f"      {rule}: {desc}")

    # ── Step 2: Create the code generation plan ────────────────────────────
    planner   = CodeGenerationPlanner(CONSTRUCTION_ROOT)
    plan_path = planner.create_plan(design_summary)

    # ── Gate: plan must be approved before code generation begins ─────────
    approved = approval_gate(
        "Have you reviewed the code-generation-plan.md and confirmed it is correct?\n"
        "  Code generation will begin immediately after approval."
    )

    if not approved:
        print(f"\n  {Fore.RED}Construction rejected.{Style.RESET_ALL}  Revise plan and re-run.")
        audit_log("construction", "phase-rejected", "Human rejected plan at gate.")
        return None

    # ── Step 3: Execute plan ───────────────────────────────────────────────
    tracker  = CheckboxTracker(plan_path)
    executor = CodeGenerationExecutor(planner, tracker, design_summary, security_enabled)
    executor.execute()

    # ── Complete ───────────────────────────────────────────────────────────
    audit_log("construction", "phase-complete",
              f"Generated files in: {CODE_OUTPUT_ROOT}")

    print(f"\n{'═'*65}")
    print(f"  ✅  CONSTRUCTION PHASE COMPLETE")
    print(f"{'═'*65}")
    print(f"  Generated code in:   {CODE_OUTPUT_ROOT}/")
    print(f"  Plan (with checkboxes): {plan_path}")
    generated = list(Path(CODE_OUTPUT_ROOT).glob("*.py")) if os.path.exists(CODE_OUTPUT_ROOT) else []
    for f in sorted(generated):
        print(f"    {f.name}")
    print(f"{'─'*65}")
    print(f"  Next step → Lesson 14: Full Adaptive Workflow (Orchestrator)")
    print(f"{'═'*65}\n")

    return {"code_dir": CODE_OUTPUT_ROOT, "plan_path": plan_path}


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_construction_phase()
