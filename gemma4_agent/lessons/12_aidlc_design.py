"""
LESSON 12 — AI-DLC: Application Design (Question-Driven)
==========================================================
"Design is iterative and question-driven — not a single model call."

Building on Lesson 11 (Inception), this lesson implements the Application
Design phase of AI-DLC.  The key insight:

  Rushing from requirements straight to code produces the wrong architecture.
  Design must interrogate the requirements through structured questions before
  generating any components, services, or dependency diagrams.

Four design categories (from AI-DLC application-design.md):
  1. Component Identification — what modules/classes exist?
  2. Component Methods       — what does each component do?
  3. Service Layer           — how do components collaborate?
  4. Dependencies            — external libraries and internal couplings

AI-DLC Steps 8 + 9 demonstrated:
  Step 8: Write design questions with [Answer]: tags
  Step 9: Parse answers; detect vague responses; add follow-up questions

Artifacts produced (all inside aidlc-docs/):
  design/application-design-plan.md          ← questions + [Answer]: slots
  design/components.md                       ← component catalogue
  design/component-methods.md                ← method signatures + descriptions
  design/services.md                         ← service layer design
  design/component-dependency.md             ← dependency map

AUTO_APPROVE = True lets the demo run with pre-filled sample answers.

Run:  python3 lessons/12_aidlc_design.py
"""

import json
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
AIDLC_ROOT   = os.path.join(PROJECT_ROOT, "aidlc-docs")
DESIGN_ROOT  = os.path.join(AIDLC_ROOT, "design")

# ── Sample requirements (used when Lesson 11 artifacts are absent) ────────────

SAMPLE_REQUIREMENTS = """
## Overview
A command-line stock portfolio summary tool for individual investors.

## Functional Requirements
- FR-01: Display current price for each stock holding from market_data.csv
- FR-02: Calculate daily P&L per holding (price change × shares)
- FR-03: Show total portfolio value in USD
- FR-04: Highlight the top gainer and top loser for the day
- FR-05: Read holdings (shares, average cost) from portfolio.db

## Non-Functional Requirements
- NFR-01: Performance — complete full summary in under 5 seconds
- NFR-02: Reliability — gracefully handle missing tickers in CSV
- NFR-03: Usability — formatted terminal output with colour coding

## Actors / Users
Individual investor running the tool from a terminal.

## Constraints
Offline only; uses local CSV + SQLite; Python 3 standard library + openai.

## Out of Scope
Live market data feeds, web UI, authentication.
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
    """Append to audit.md (mirrors Lesson 11 AuditLogger)."""
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


def load_requirements() -> str:
    """Load requirements from Lesson 11 output or fall back to sample."""
    req_path = os.path.join(AIDLC_ROOT, "inception", "requirements", "requirements.md")
    if os.path.exists(req_path):
        with open(req_path) as f:
            return f.read()
    print(f"  {Fore.YELLOW}[INFO]{Style.RESET_ALL}  "
          f"No Lesson 11 requirements found — using built-in sample.")
    return SAMPLE_REQUIREMENTS


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — DesignQuestionWriter (AI-DLC Step 8)
# ══════════════════════════════════════════════════════════════════════════════

DESIGN_QUESTION_SYSTEM = """You are an AI-DLC Application Design Analyst.

Given a requirements document, write structured design questions across
four categories.  Each question must have a blank [Answer]: tag.

Format exactly as:
### Component Identification
**DQ-CI-01**: What are the top-level components/classes in this system?
[Answer]: 

**DQ-CI-02**: Which components are stateless vs stateful?
[Answer]: 

### Component Methods
**DQ-CM-01**: What is the primary method/operation of each component?
[Answer]: 

**DQ-CM-02**: What data does each component accept as input and return?
[Answer]: 

### Service Layer
**DQ-SL-01**: How do components communicate — direct calls or via a coordinator?
[Answer]: 

**DQ-SL-02**: Is there a single orchestrator or do components self-orchestrate?
[Answer]: 

### Dependencies
**DQ-DEP-01**: What external libraries are needed?
[Answer]: 

**DQ-DEP-02**: Which internal components depend on which other components?
[Answer]: 

Rules:
- Tailor questions to the specific requirements given — do not be generic
- Leave all [Answer]: lines blank
- Output ONLY the questions — no preamble, no summary
"""


class DesignQuestionWriter:
    """
    AI-DLC Step 8: Write structured design questions with [Answer]: tags.

    Creates aidlc-docs/design/application-design-plan.md.
    The format mirrors exactly what AI-DLC specifies: category headers,
    numbered questions, and blank [Answer]: slots.
    """

    SAMPLE_ANSWERS = {
        "DQ-CI-01": "PortfolioReader, MarketDataReader, PnLCalculator, SummaryPrinter",
        "DQ-CI-02": "All components are stateless; data is passed as arguments",
        "DQ-CM-01": "PortfolioReader.load() returns list of holdings; MarketDataReader.get(ticker) returns price dict; PnLCalculator.compute(holdings, prices) returns enriched list; SummaryPrinter.render(enriched) prints formatted table",
        "DQ-CM-02": "PortfolioReader: no args → list[dict]. MarketDataReader: ticker str → dict. PnLCalculator: list, dict → list. SummaryPrinter: list → None (prints to stdout)",
        "DQ-SL-01": "Direct sequential calls — orchestrator calls each component in order",
        "DQ-SL-02": "Single orchestrator function: run_summary() coordinates all components",
        "DQ-DEP-01": "csv (stdlib), sqlite3 (stdlib), colorama (already in requirements.txt)",
        "DQ-DEP-02": "run_summary → PortfolioReader, MarketDataReader, PnLCalculator, SummaryPrinter; PnLCalculator depends on outputs from the two readers",
    }

    def __init__(self, docs_root: str):
        self.out_path = os.path.join(docs_root, "application-design-plan.md")

    def write(self, requirements: str) -> str:
        print(f"\n{Fore.BLUE}▶ DesignQuestionWriter{Style.RESET_ALL}  writing design questions…")
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

        questions = ask_model(DESIGN_QUESTION_SYSTEM, requirements)

        header = (
            "# Application Design Plan\n\n"
            "*AI-DLC Step 8 — Design questions generated from requirements.*\n\n"
            "Fill in every [Answer]: field before proceeding.\n\n"
            "---\n\n"
        )
        with open(self.out_path, "w") as f:
            f.write(header + questions + "\n")

        audit_log("design", "design-questions-written", f"Written to: {self.out_path}")
        print(f"    Written → {self.out_path}")
        return self.out_path

    def fill_sample_answers(self):
        """Inject sample answers (AUTO_APPROVE mode)."""
        if not AUTO_APPROVE:
            print(f"\n  Please fill in [Answer]: fields in:\n  {self.out_path}")
            input("  Press Enter when done…")
            return

        print(f"  {Fore.YELLOW}[AUTO-APPROVE]{Style.RESET_ALL}  Injecting sample design answers…")
        with open(self.out_path) as f:
            content = f.read()

        for tag, answer in self.SAMPLE_ANSWERS.items():
            # Find the question block for this tag and fill the next [Answer]:
            content = re.sub(
                rf"(\*\*{re.escape(tag)}\*\*.*?\n\[Answer\]): ",
                rf"\1: {answer}",
                content,
                flags=re.DOTALL,
            )

        with open(self.out_path, "w") as f:
            f.write(content)

        audit_log("design", "design-answers-filled", "(sample answers — AUTO_APPROVE mode)")


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — AnswerParser (AI-DLC Step 9)
# ══════════════════════════════════════════════════════════════════════════════

VAGUE_DETECTOR_SYSTEM = """You are a design-answer quality reviewer.

Given a list of design question + answer pairs, identify any answers that are
vague, too short, or ambiguous.  For each vague answer, output a follow-up
question that would clarify it.

Output JSON:
{
  "issues": [
    {"tag": "DQ-CI-01", "problem": "too vague", "followup": "...?"},
    ...
  ]
}

If all answers are clear, output: {"issues": []}

Vague indicators: fewer than 5 words, "I don't know", "TBD", "maybe", "various".
"""


class AnswerParser:
    """
    AI-DLC Step 9: Parse answers, detect vague responses, add follow-ups.

    Reads the completed application-design-plan.md, extracts all [Answer]:
    values, sends them to Gemma4 for quality review.  If any answer is
    detected as vague, follow-up questions are appended to the document.
    """

    def __init__(self, plan_path: str):
        self.plan_path = plan_path

    def parse(self) -> dict[str, str]:
        """Extract all tag → answer mappings from the plan document."""
        with open(self.plan_path) as f:
            content = f.read()

        answers = {}
        # Find all DQ-* tags followed by [Answer]: value
        for match in re.finditer(
            r"\*\*(DQ-[A-Z]+-\d+)\*\*.*?\n\[Answer\]:\s*(.+)",
            content,
            re.DOTALL,
        ):
            tag    = match.group(1)
            answer = match.group(2).strip()
            answers[tag] = answer

        return answers

    def check_quality(self, answers: dict[str, str]) -> list[dict]:
        """Ask Gemma4 to identify vague answers and generate follow-ups."""
        print(f"\n{Fore.BLUE}▶ AnswerParser{Style.RESET_ALL}  checking answer quality…")

        qa_pairs = "\n".join(
            f"{tag}: {ans}"
            for tag, ans in answers.items()
        )

        raw = ask_model(VAGUE_DETECTOR_SYSTEM, qa_pairs)

        # Parse JSON from response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return []
        try:
            result = json.loads(match.group())
            issues = result.get("issues", [])
        except Exception:
            issues = []

        if issues:
            print(f"    {Fore.YELLOW}Found {len(issues)} vague answer(s) — appending follow-ups{Style.RESET_ALL}")
            self._append_followups(issues)
        else:
            print(f"    {Fore.GREEN}All answers are clear — no follow-ups needed{Style.RESET_ALL}")

        return issues

    def _append_followups(self, issues: list[dict]):
        followup_text = "\n\n---\n\n## Follow-up Questions\n\n"
        followup_text += "*AI-DLC Step 9 — clarify vague answers before proceeding.*\n\n"
        for issue in issues:
            tag      = issue.get("tag", "?")
            problem  = issue.get("problem", "vague")
            followup = issue.get("followup", "Please elaborate.")
            followup_text += (
                f"**{tag} Follow-up** ({problem}):\n"
                f"{followup}\n"
                f"[Answer]: (please fill in)\n\n"
            )
        with open(self.plan_path, "a") as f:
            f.write(followup_text)
        audit_log("design", "followup-questions-appended",
                  f"{len(issues)} follow-up(s) added to design plan.")


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — ComponentDesigner
# ══════════════════════════════════════════════════════════════════════════════

COMPONENTS_SYSTEM = """You are an AI-DLC Component Designer.

Given requirements and design answers, generate a components.md document.

Format:
# Components

## {ComponentName}
**Purpose**: one sentence
**Responsibility**: what it owns / manages
**Interface**: public methods it exposes (names only at this point)

Rules:
- One ## section per component
- Be specific to the requirements — do not invent generic utilities
- Output ONLY the document
"""

COMPONENT_METHODS_SYSTEM = """You are an AI-DLC Component Methods Designer.

Given requirements and component descriptions, generate component-methods.md.

Format:
# Component Methods

## {ComponentName}

### method_name(param: type, ...) -> return_type
**Purpose**: what it does
**Input**: describe each parameter
**Output**: describe the return value
**Error handling**: what exceptions it may raise

Rules:
- Cover every public method of every component
- Use Python type hints
- Output ONLY the document
"""


class ComponentDesigner:
    """
    Generates components.md and component-methods.md from requirements + answers.

    AI-DLC separates component identification (what exists) from method design
    (what each component does in detail).  Both documents feed Construction.
    """

    def __init__(self, docs_root: str):
        self.components_path = os.path.join(docs_root, "components.md")
        self.methods_path    = os.path.join(docs_root, "component-methods.md")

    def generate_components(self, requirements: str, answers: dict[str, str]) -> str:
        print(f"\n{Fore.BLUE}▶ ComponentDesigner{Style.RESET_ALL}  generating components.md…")

        context = (
            f"Requirements:\n{requirements}\n\n"
            f"Design answers:\n" +
            "\n".join(f"{tag}: {ans}" for tag, ans in answers.items())
        )
        content = ask_model(COMPONENTS_SYSTEM, context)

        with open(self.components_path, "w") as f:
            f.write("# Components\n\n")
            f.write(f"*Generated by AI-DLC Design — {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}*\n\n")
            f.write("---\n\n")
            f.write(content)

        audit_log("design", "components-generated", f"Written to: {self.components_path}")
        print(f"    Written → {self.components_path}")
        return self.components_path

    def generate_methods(self, requirements: str, components_text: str) -> str:
        print(f"\n{Fore.BLUE}▶ ComponentDesigner{Style.RESET_ALL}  generating component-methods.md…")

        context = (
            f"Requirements:\n{requirements}\n\n"
            f"Components:\n{components_text}"
        )
        content = ask_model(COMPONENT_METHODS_SYSTEM, context)

        with open(self.methods_path, "w") as f:
            f.write("# Component Methods\n\n")
            f.write(f"*Generated by AI-DLC Design — {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}*\n\n")
            f.write("---\n\n")
            f.write(content)

        audit_log("design", "component-methods-generated", f"Written to: {self.methods_path}")
        print(f"    Written → {self.methods_path}")
        return self.methods_path


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — ServiceDesigner
# ══════════════════════════════════════════════════════════════════════════════

SERVICES_SYSTEM = """You are an AI-DLC Service Layer Designer.

Given requirements and component descriptions, generate services.md.

Format:
# Services

## {ServiceName}
**Type**: orchestrator | data-access | transformation | presentation
**Purpose**: one sentence
**Collaborates with**: {ComponentA}, {ComponentB}
**Sequence**:
1. ...
2. ...

Rules:
- Describe how components are coordinated
- Include the main orchestrator service (if any)
- Output ONLY the document
"""

DEPENDENCY_SYSTEM = """You are an AI-DLC Dependency Mapper.

Given requirements, components, and services, generate component-dependency.md.

Format:
# Component Dependencies

## External Libraries
| Library | Used by | Purpose |
|---------|---------|---------|
| ...     | ...     | ...     |

## Internal Dependencies
| Component | Depends on | Reason |
|-----------|------------|--------|
| ...       | ...        | ...    |

Rules:
- Cover both external (pip) and internal (component-to-component) dependencies
- Be specific to the requirements
- Output ONLY the document
"""


class ServiceDesigner:
    """
    Generates services.md and component-dependency.md.

    The service layer document is critical for understanding how to implement
    the orchestrator in Construction.  The dependency map prevents circular
    import problems before any code is written.
    """

    def __init__(self, docs_root: str):
        self.services_path = os.path.join(docs_root, "services.md")
        self.deps_path     = os.path.join(docs_root, "component-dependency.md")

    def generate_services(self, requirements: str, components_text: str) -> str:
        print(f"\n{Fore.BLUE}▶ ServiceDesigner{Style.RESET_ALL}  generating services.md…")

        context = f"Requirements:\n{requirements}\n\nComponents:\n{components_text}"
        content = ask_model(SERVICES_SYSTEM, context)

        with open(self.services_path, "w") as f:
            f.write("# Services\n\n")
            f.write(f"*Generated by AI-DLC Design — {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}*\n\n")
            f.write("---\n\n")
            f.write(content)

        audit_log("design", "services-generated", f"Written to: {self.services_path}")
        print(f"    Written → {self.services_path}")
        return self.services_path

    def generate_dependencies(
        self,
        requirements: str,
        components_text: str,
        services_text: str,
    ) -> str:
        print(f"\n{Fore.BLUE}▶ ServiceDesigner{Style.RESET_ALL}  generating component-dependency.md…")

        context = (
            f"Requirements:\n{requirements}\n\n"
            f"Components:\n{components_text}\n\n"
            f"Services:\n{services_text}"
        )
        content = ask_model(DEPENDENCY_SYSTEM, context)

        with open(self.deps_path, "w") as f:
            f.write("# Component Dependencies\n\n")
            f.write(f"*Generated by AI-DLC Design — {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}*\n\n")
            f.write("---\n\n")
            f.write(content)

        audit_log("design", "dependencies-generated", f"Written to: {self.deps_path}")
        print(f"    Written → {self.deps_path}")
        return self.deps_path


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — run_design_phase
# ══════════════════════════════════════════════════════════════════════════════

def run_design_phase(requirements: str | None = None):
    """
    Run the full AI-DLC Application Design phase.

    Steps:
      1. Load requirements (Lesson 11 output or built-in sample)
      2. Write design questions → fill answers
      3. Parse + quality-check answers; detect vague responses
      4. Generate components.md + component-methods.md
      5. Generate services.md + component-dependency.md
      6. Approval gate
    """
    print(f"\n{'═'*65}")
    print(f"  AI-DLC APPLICATION DESIGN PHASE")
    print(f"{'═'*65}")

    if requirements is None:
        requirements = load_requirements()

    os.makedirs(DESIGN_ROOT, exist_ok=True)
    audit_log("design", "phase-started")

    # ── Step 1: Write design questions ────────────────────────────────────
    question_writer = DesignQuestionWriter(DESIGN_ROOT)
    plan_path       = question_writer.write(requirements)
    question_writer.fill_sample_answers()

    # ── Step 2: Parse + quality-check answers ─────────────────────────────
    parser  = AnswerParser(plan_path)
    answers = parser.parse()
    issues  = parser.check_quality(answers)

    print(f"\n  Parsed answers: {len(answers)}  |  Issues found: {len(issues)}")

    # ── Step 3: Generate component documents ──────────────────────────────
    component_designer = ComponentDesigner(DESIGN_ROOT)
    comp_path  = component_designer.generate_components(requirements, answers)

    with open(comp_path) as f:
        components_text = f.read()

    methods_path = component_designer.generate_methods(requirements, components_text)

    # ── Step 4: Generate service + dependency documents ───────────────────
    service_designer = ServiceDesigner(DESIGN_ROOT)
    svc_path  = service_designer.generate_services(requirements, components_text)

    with open(svc_path) as f:
        services_text = f.read()

    deps_path = service_designer.generate_dependencies(requirements, components_text, services_text)

    # ── Approval gate ──────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  📐  Design artifacts written to: {DESIGN_ROOT}/")
    print(f"      application-design-plan.md")
    print(f"      components.md")
    print(f"      component-methods.md")
    print(f"      services.md")
    print(f"      component-dependency.md")

    approved = approval_gate(
        "Have you reviewed all design documents and confirmed the architecture is correct?"
    )

    if not approved:
        print(f"\n  {Fore.RED}Design phase rejected.{Style.RESET_ALL}  Revise and re-run.")
        audit_log("design", "phase-rejected", "Human rejected at approval gate.")
        return None

    audit_log("design", "phase-complete", "All design artifacts produced and approved.")

    print(f"\n{'═'*65}")
    print(f"  ✅  DESIGN PHASE COMPLETE")
    print(f"{'═'*65}")
    print(f"  Next step → Lesson 13: Construction (plan-then-generate)")
    print(f"{'═'*65}\n")

    return {
        "components_path":   comp_path,
        "methods_path":      methods_path,
        "services_path":     svc_path,
        "dependencies_path": deps_path,
    }


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_design_phase()
