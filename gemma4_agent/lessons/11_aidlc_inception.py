"""
LESSON 11 — AI-DLC: The Inception Phase
========================================
"Think before you code — determine WHAT and WHY before any HOW."

AI-DLC (AI-Driven Development Lifecycle) is a model-agnostic methodology.
The rules live in structured markdown prompts; Gemma4 executes them.

This lesson implements the Inception phase — the first gate in AI-DLC.
No code is written until:
  1. The request is classified (intent, scope, complexity)
  2. Clarifying questions are asked and answered
  3. Requirements are formalised into a requirements.md document
  4. The workflow plan (which downstream stages to run) is decided
  5. A human approves the requirements

Key AI-DLC principle demonstrated here:
  "Determine WHAT and WHY before any HOW."

Artifacts produced (all inside aidlc-docs/):
  audit.md                                    ← every interaction logged
  aidlc-state.md                              ← stage progress tracker
  inception/requirements/
    requirement-verification-questions.md     ← clarifying questions (with [Answer]: slots)
    requirements.md                           ← final formalised requirements
  inception/
    workflow-plan.md                          ← which stages will run and why

AUTO_APPROVE = True   → demo mode (runs without blocking for user input)
AUTO_APPROVE = False  → real interactive mode (waits at each gate)

Run:  python3 lessons/11_aidlc_inception.py
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

AUTO_APPROVE = True   # set False for real interactive approval gates

MODEL = "docker.io/ai/gemma4:E4B"

client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="not-needed",
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
AIDLC_ROOT   = os.path.join(PROJECT_ROOT, "aidlc-docs")


# ══════════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE — AuditLogger + StateTracker
# ══════════════════════════════════════════════════════════════════════════════

class AuditLogger:
    """
    Appends every significant interaction to aidlc-docs/audit.md.

    Each entry is ISO-8601 timestamped.  The audit log is append-only —
    never overwritten — so you have a full record of every decision made.

    AI-DLC requirement: all significant decisions must be traceable.
    """

    def __init__(self, audit_path: str):
        self.path = audit_path
        os.makedirs(os.path.dirname(audit_path), exist_ok=True)
        if not os.path.exists(audit_path):
            self._write_header()

    def _write_header(self):
        ts = self._ts()
        with open(self.path, "w") as f:
            f.write(f"# AI-DLC Audit Log\n\nInitialised: {ts}\n\n")

    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def log(self, stage: str, event: str, detail: str = ""):
        entry = f"## [{self._ts()}] {stage} — {event}\n"
        if detail:
            entry += f"\n{detail.strip()}\n"
        entry += "\n---\n\n"
        with open(self.path, "a") as f:
            f.write(entry)
        print(f"  {Fore.CYAN}[AUDIT]{Style.RESET_ALL}  {stage} → {event}")


class StateTracker:
    """
    Reads/writes aidlc-docs/aidlc-state.md — a simple stage-progress file.

    Each stage has a status: pending / in-progress / complete / skipped.
    The file is human-readable markdown so teams can inspect it directly.

    AI-DLC requirement: always know where you are in the lifecycle.
    """

    STAGES = [
        "inception",
        "application-design",
        "construction",
        "build-and-test",
    ]

    def __init__(self, state_path: str):
        self.path = state_path
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        if not os.path.exists(state_path):
            self._init_file()

    def _init_file(self):
        lines = ["# AI-DLC State\n\n"]
        for stage in self.STAGES:
            lines.append(f"- [ ] **{stage}**: pending\n")
        with open(self.path, "w") as f:
            f.writelines(lines)

    def _read(self) -> str:
        with open(self.path) as f:
            return f.read()

    def _write(self, content: str):
        with open(self.path, "w") as f:
            f.write(content)

    def set_status(self, stage: str, status: str):
        """Update a stage status: pending / in-progress / complete / skipped."""
        content = self._read()
        # Replace the line for this stage
        marker = "x" if status == "complete" else " "
        content = re.sub(
            rf"- \[[ x]\] \*\*{re.escape(stage)}\*\*:.*",
            f"- [{marker}] **{stage}**: {status}",
            content,
        )
        self._write(content)
        print(f"  {Fore.MAGENTA}[STATE]{Style.RESET_ALL}   {stage} → {status}")

    def get_status(self, stage: str) -> str:
        content = self._read()
        match = re.search(rf"\*\*{re.escape(stage)}\*\*: (\S+)", content)
        return match.group(1) if match else "unknown"

    def add_custom(self, key: str, value: str):
        """Append a custom key-value annotation (e.g. complexity=standard)."""
        with open(self.path, "a") as f:
            f.write(f"\n**{key}**: {value}\n")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — one-shot model call
# ══════════════════════════════════════════════════════════════════════════════

def ask_model(system: str, user: str) -> str:
    """Call Gemma4, strip <think>…</think>, return clean text."""
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


def approval_gate(message: str) -> bool:
    """
    Approval gate — AI-DLC requires human confirmation before proceeding.
    In AUTO_APPROVE mode the gate always passes (useful for demo runs).
    """
    print(f"\n{'─'*65}")
    print(f"  ⛔  GATE — DO NOT PROCEED until confirmed")
    print(f"  {message}")
    print(f"{'─'*65}")
    if AUTO_APPROVE:
        print(f"  {Fore.YELLOW}[AUTO-APPROVE]{Style.RESET_ALL}  Gate passed automatically (AUTO_APPROVE=True)")
        return True
    answer = input("  Approve? (y/n): ").strip().lower()
    return answer == "y"


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — IntentAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

INTENT_SYSTEM = """You are an AI-DLC Intent Analyzer. Given a user's request, classify it.

Output a JSON object with exactly these fields (no markdown, no prose):
{
  "request_type": "new_feature" | "bug_fix" | "refactor" | "research" | "new_project",
  "scope":        "single_module" | "multi_module" | "full_application",
  "complexity":   "low" | "medium" | "high",
  "depth_needed": "minimal" | "standard" | "comprehensive",
  "summary":      "one sentence describing what is being built"
}

depth_needed rules:
  minimal       → trivial change, no design needed
  standard      → typical feature, standard inception + design
  comprehensive → cross-cutting concern, full lifecycle required
"""


class IntentAnalyzer:
    """
    Step 1 of AI-DLC Inception: classify the user's request.

    Gemma4 reads the request and outputs a structured classification.
    This classification drives all downstream decisions (which stages run,
    how deep the requirements analysis goes, etc.).
    """

    def __init__(self, audit: AuditLogger, state: StateTracker):
        self.audit = audit
        self.state = state

    def analyze(self, request: str) -> dict:
        print(f"\n{Fore.BLUE}▶ IntentAnalyzer{Style.RESET_ALL}  classifying request…")
        raw = ask_model(INTENT_SYSTEM, f"Request: {request}")

        # Extract JSON from the response
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            # Fallback classification
            intent = {
                "request_type": "new_project",
                "scope":        "full_application",
                "complexity":   "medium",
                "depth_needed": "standard",
                "summary":      request[:100],
            }
        else:
            try:
                import json
                intent = json.loads(match.group())
            except Exception:
                intent = {
                    "request_type": "new_project",
                    "scope":        "full_application",
                    "complexity":   "medium",
                    "depth_needed": "standard",
                    "summary":      request[:100],
                }

        self.audit.log("inception", "intent-analyzed", f"```json\n{intent}\n```")
        self.state.add_custom("intent.complexity",   intent.get("complexity",   "medium"))
        self.state.add_custom("intent.scope",        intent.get("scope",        "full_application"))
        self.state.add_custom("intent.depth_needed", intent.get("depth_needed", "standard"))

        print(f"    complexity={intent.get('complexity')}  "
              f"scope={intent.get('scope')}  "
              f"depth={intent.get('depth_needed')}")
        return intent


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — ClarifyingQuestionsAgent
# ══════════════════════════════════════════════════════════════════════════════

CLARIFYING_SYSTEM = """You are an AI-DLC Requirements Analyst.

Your job is to write clarifying questions that will unambiguously define the
requirements for a software project.  Write exactly 5 questions.

Format each question as:
**Q{n}: {question text}**
[Answer]: 

Rules:
- Questions must be specific, not generic ("what tech stack?" is good; "any other requirements?" is bad)
- Cover: users/actors, key functionality, data/storage, constraints/NFRs, integration points
- Leave the [Answer]: line blank — a human will fill it in
- Output ONLY the 5 questions. No intro, no summary.
"""


class ClarifyingQuestionsAgent:
    """
    Step 2 of AI-DLC Inception: write structured clarifying questions.

    Gemma4 generates 5 targeted questions and writes them to:
      aidlc-docs/inception/requirements/requirement-verification-questions.md

    Each question has an [Answer]: tag.  In real mode the user fills these in.
    In AUTO_APPROVE mode we inject sample answers so the demo runs end-to-end.
    """

    SAMPLE_ANSWERS = {
        1: "The primary user is an individual investor who wants a daily summary of their stock portfolio.",
        2: "Key functionality: show current price, daily P&L, total portfolio value, and top gainer/loser.",
        3: "Data will be fetched from the local market_data.csv and portfolio.db (already present in the project).",
        4: "Must run in under 5 seconds; no authentication needed; output as formatted terminal text.",
        5: "No external integrations required; must work offline using local data files only.",
    }

    def __init__(self, audit: AuditLogger, docs_root: str):
        self.audit     = audit
        self.docs_root = docs_root
        self.out_path  = os.path.join(
            docs_root, "inception", "requirements",
            "requirement-verification-questions.md"
        )

    def write_questions(self, request: str, intent: dict) -> str:
        print(f"\n{Fore.BLUE}▶ ClarifyingQuestionsAgent{Style.RESET_ALL}  writing questions…")
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

        context = (
            f"Request: {request}\n"
            f"Type: {intent.get('request_type')}  "
            f"Scope: {intent.get('scope')}  "
            f"Complexity: {intent.get('complexity')}"
        )
        questions_text = ask_model(CLARIFYING_SYSTEM, context)

        # Build the full document
        header = (
            "# Requirement Verification Questions\n\n"
            f"**Request:** {request}\n\n"
            "---\n\n"
        )
        with open(self.out_path, "w") as f:
            f.write(header + questions_text + "\n")

        self.audit.log("inception", "clarifying-questions-written", f"Written to: {self.out_path}")
        print(f"    Written → {self.out_path}")
        return self.out_path

    def fill_sample_answers(self):
        """
        In AUTO_APPROVE mode: inject sample answers into the questions file.
        In real mode a human would edit the file and we'd re-read it.
        """
        if not AUTO_APPROVE:
            print(f"\n  Please fill in the [Answer]: fields in:\n  {self.out_path}")
            input("  Press Enter when done…")
            return

        print(f"  {Fore.YELLOW}[AUTO-APPROVE]{Style.RESET_ALL}  Injecting sample answers…")
        with open(self.out_path) as f:
            content = f.read()

        # Replace each [Answer]:  with a sample answer
        for i, answer in self.SAMPLE_ANSWERS.items():
            # Replace the i-th blank [Answer]:
            content = content.replace("[Answer]: \n", f"[Answer]: {answer}\n", 1)

        with open(self.out_path, "w") as f:
            f.write(content)

        self.audit.log("inception", "answers-filled", "(sample answers — AUTO_APPROVE mode)")

    def read_answers(self) -> dict[int, str]:
        """Parse [Answer]: tags from the questions file."""
        with open(self.out_path) as f:
            content = f.read()

        answers = {}
        for i, match in enumerate(re.finditer(r"\[Answer\]:\s*(.+)", content), start=1):
            answers[i] = match.group(1).strip()
        return answers


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — RequirementsAgent
# ══════════════════════════════════════════════════════════════════════════════

REQUIREMENTS_SYSTEM = """You are an AI-DLC Requirements Engineer.

Given a user request and a set of answered clarifying questions, write a
formal requirements document in markdown.

Structure:
## Overview
(1–2 sentence project summary)

## Functional Requirements
- FR-01: ...
- FR-02: ...
(list every distinct feature; at least 4)

## Non-Functional Requirements
- NFR-01: Performance — ...
- NFR-02: Reliability — ...
(at least 3)

## Actors / Users
(who uses the system)

## Constraints
(technical constraints, scope limits)

## Out of Scope
(what will NOT be built)

Rules:
- Be specific and testable ("display total portfolio value in USD" not "show portfolio")
- Number every requirement (FR-01, FR-02, …)
- Output ONLY the document — no preamble
"""


class RequirementsAgent:
    """
    Step 3 of AI-DLC Inception: generate formal requirements.

    Reads the answered questions file and generates requirements.md.
    The requirements document drives all subsequent design and construction.
    """

    def __init__(self, audit: AuditLogger, docs_root: str):
        self.audit    = audit
        self.out_path = os.path.join(
            docs_root, "inception", "requirements", "requirements.md"
        )

    def generate(self, request: str, answers: dict[int, str]) -> str:
        print(f"\n{Fore.BLUE}▶ RequirementsAgent{Style.RESET_ALL}  generating requirements.md…")

        answers_text = "\n".join(f"A{i}: {a}" for i, a in answers.items())
        user_msg = f"Request: {request}\n\nAnswers:\n{answers_text}"

        requirements = ask_model(REQUIREMENTS_SYSTEM, user_msg)

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, "w") as f:
            f.write("# Requirements\n\n")
            f.write(f"*Generated by AI-DLC Inception — {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}*\n\n")
            f.write("---\n\n")
            f.write(requirements)
            f.write("\n")

        self.audit.log("inception", "requirements-generated", f"Written to: {self.out_path}")
        print(f"    Written → {self.out_path}")
        return self.out_path


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — WorkflowPlanner
# ══════════════════════════════════════════════════════════════════════════════

WORKFLOW_SYSTEM = """You are an AI-DLC Workflow Planner.

Given the intent classification (complexity, scope, depth_needed), decide
which AI-DLC stages to run.

AI-DLC stage rules:
  ALWAYS run:
    - inception         (you just completed it)
    - construction      (always required for code)

  CONDITIONAL:
    - application-design:  run if complexity=medium|high OR scope=multi_module|full_application
    - build-and-test:      run if complexity=high OR scope=full_application

Output a markdown document listing each stage with a ✅ (run) or ⏭ (skip)
and a one-sentence justification.  No prose outside the table.

Format:
## Workflow Plan

| Stage | Decision | Reason |
|-------|----------|--------|
| inception | ✅ Run | ... |
| application-design | ✅ Run / ⏭ Skip | ... |
| construction | ✅ Run | ... |
| build-and-test | ✅ Run / ⏭ Skip | ... |
"""


class WorkflowPlanner:
    """
    Step 4 of AI-DLC Inception: decide which downstream stages to execute.

    AI-DLC is adaptive — not every project needs every stage.
    A simple single-function change skips application-design.
    A full application runs all stages.

    This is the ALWAYS / CONDITIONAL logic from AI-DLC core-workflow.md.
    """

    def __init__(self, audit: AuditLogger, docs_root: str):
        self.audit    = audit
        self.out_path = os.path.join(docs_root, "inception", "workflow-plan.md")

    def plan(self, intent: dict) -> dict[str, bool]:
        print(f"\n{Fore.BLUE}▶ WorkflowPlanner{Style.RESET_ALL}  deciding which stages to run…")

        complexity   = intent.get("complexity",   "medium")
        scope        = intent.get("scope",        "full_application")
        depth_needed = intent.get("depth_needed", "standard")

        context = (
            f"complexity={complexity}\n"
            f"scope={scope}\n"
            f"depth_needed={depth_needed}"
        )
        plan_text = ask_model(WORKFLOW_SYSTEM, context)

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, "w") as f:
            f.write(f"# Workflow Plan\n\n")
            f.write(f"*Intent: complexity={complexity}, scope={scope}, depth={depth_needed}*\n\n")
            f.write("---\n\n")
            f.write(plan_text)
            f.write("\n")

        # Derive boolean flags from the plan text
        stages = {
            "application-design": (
                complexity in ("medium", "high") or
                scope in ("multi_module", "full_application")
            ),
            "build-and-test": (
                complexity == "high" or
                scope == "full_application"
            ),
        }

        self.audit.log("inception", "workflow-planned", f"Written to: {self.out_path}")
        print(f"    Written → {self.out_path}")
        for stage, run in stages.items():
            icon = "✅" if run else "⏭"
            print(f"    {icon}  {stage}")

        return stages


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — run_inception_phase
# ══════════════════════════════════════════════════════════════════════════════

def run_inception_phase(request: str):
    """
    Run the full AI-DLC Inception phase for a given request.

    Steps:
      1. Initialise infrastructure (AuditLogger, StateTracker)
      2. Analyze intent
      3. Write clarifying questions → fill answers
      4. Generate requirements
      5. Plan the workflow
      6. Approval gate
    """
    print(f"\n{'═'*65}")
    print(f"  AI-DLC INCEPTION PHASE")
    print(f"{'═'*65}")
    print(f"  Request: {request}")
    print(f"{'─'*65}")

    # ── Initialise infrastructure ──────────────────────────────────────────
    audit_path = os.path.join(AIDLC_ROOT, "audit.md")
    state_path = os.path.join(AIDLC_ROOT, "aidlc-state.md")

    audit = AuditLogger(audit_path)
    state = StateTracker(state_path)

    audit.log("inception", "phase-started", f"Request: {request}")
    state.set_status("inception", "in-progress")

    # ── Step 1: Analyze intent ─────────────────────────────────────────────
    analyzer = IntentAnalyzer(audit, state)
    intent   = analyzer.analyze(request)

    # ── Step 2: Clarifying questions ───────────────────────────────────────
    questions_agent = ClarifyingQuestionsAgent(audit, AIDLC_ROOT)
    questions_agent.write_questions(request, intent)
    questions_agent.fill_sample_answers()
    answers = questions_agent.read_answers()

    print(f"\n  Answers received: {len(answers)}")
    for i, a in answers.items():
        print(f"    A{i}: {a[:80]}{'…' if len(a) > 80 else ''}")

    # ── Step 3: Generate requirements ─────────────────────────────────────
    req_agent = RequirementsAgent(audit, AIDLC_ROOT)
    req_path  = req_agent.generate(request, answers)

    # ── Step 4: Plan the workflow ──────────────────────────────────────────
    planner       = WorkflowPlanner(audit, AIDLC_ROOT)
    workflow_plan = planner.plan(intent)

    # ── Step 5: Approval gate ──────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  📄  Requirements written to:")
    print(f"      {req_path}")
    print(f"\n  Please review the requirements document before proceeding.")

    approved = approval_gate(
        "Have you reviewed requirements.md and confirmed it is correct?"
    )

    if not approved:
        print(f"\n  {Fore.RED}Inception rejected.{Style.RESET_ALL}  "
              f"Revise requirements.md and re-run.")
        state.set_status("inception", "pending")
        audit.log("inception", "phase-rejected", "Human rejected at approval gate.")
        return None

    # ── Complete ───────────────────────────────────────────────────────────
    state.set_status("inception", "complete")
    audit.log("inception", "phase-complete", "All inception artifacts produced.")

    print(f"\n{'═'*65}")
    print(f"  ✅  INCEPTION COMPLETE")
    print(f"{'═'*65}")
    print(f"  Artifacts in: {AIDLC_ROOT}/")
    print(f"    audit.md")
    print(f"    aidlc-state.md")
    print(f"    inception/requirements/requirement-verification-questions.md")
    print(f"    inception/requirements/requirements.md")
    print(f"    inception/workflow-plan.md")
    print(f"{'─'*65}")
    print(f"  Workflow plan:")
    for stage, run in workflow_plan.items():
        icon = "✅" if run else "⏭ (skip)"
        print(f"    {icon}  {stage}")
    print(f"{'═'*65}\n")

    return {
        "intent":        intent,
        "answers":       answers,
        "requirements":  req_path,
        "workflow_plan": workflow_plan,
    }


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_inception_phase(
        "Using AI-DLC, build a stock portfolio summary tool that shows "
        "each holding's current price, daily P&L, and the overall portfolio value."
    )
