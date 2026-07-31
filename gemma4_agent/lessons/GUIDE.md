# AI Agent Curriculum — Gemma 4:E4B on Docker Model Runner

A ground-up, framework-free guide to building AI agents in Python.
Every lesson is a single, self-contained file you can run with `python3 lessons/XX_name.py`.

---

## What is an AI Agent?

A plain chat model takes a message and returns a message. That's it — one shot.

An **agent** is different. It runs in a **loop**:

```
User Goal
    │
    ▼
┌─────────────────────────────────────────┐
│  1. PERCEIVE  — read input + history    │
│  2. REASON    — LLM decides next step   │
│  3. ACT       — call a tool / function  │
│  4. OBSERVE   — inject tool result back │
│  5. REPEAT    — until task is done      │
└─────────────────────────────────────────┘
    │
    ▼
Final Answer
```

The model never "executes" anything itself. **You** run the tools and feed results back
into the conversation. The model just reasons about what to do next.

---

## The Message History IS the Agent's Brain

Everything the agent knows lives in the `messages` list:

```python
messages = [
    {"role": "system",    "content": "You are a ..."},  # personality + rules
    {"role": "user",      "content": "What is ..."},    # user request
    {"role": "assistant", "content": "I will ..."},     # model's last output
    {"role": "user",      "content": "Observation: ..."}, # tool result fed back
    # ^ this cycle repeats until Final Answer
]
```

Longer history = more context = smarter decisions, but also more tokens.

---

## Two Styles of Tool Calling

### Style 1 — ReAct (Text Parsing)
The model writes `Action: {"tool": "...", "args": {...}}` in plain text.
You parse it with string splitting or regex.

**Pros:** Simple, works with any model.  
**Cons:** Fragile — the model might misformat the JSON.

### Style 2 — Native Tool Calling (OpenAI `tools` API)
You pass a `tools=[...]` schema to the API. The model returns a structured
`tool_calls` object (not text). No parsing needed.

**Pros:** Reliable, structured, the "industry standard" way.  
**Cons:** Model must support the `tools` parameter (Gemma 4 does).

---

## Lesson Map

### Part 1 — Building Agents

| File | What you learn |
|------|----------------|
| `01_react_loop.py` | The ReAct loop — Reason, Act, Observe in plain Python |
| `02_native_tools.py` | Native tool calling with OpenAI `tools` schema |
| `03_memory.py` | Short-term (conversation) + long-term (file) memory |
| `04_multi_tool.py` | Agent that chains multiple tool calls to answer one question |
| `05_planning.py` | Plan-and-Execute — make a plan first, then execute each step |
| `06_multi_agent.py` | Orchestrator delegates tasks to specialized sub-agents |

### Part 2 — Harness Engineering

| File | What you learn |
|------|----------------|
| `07_tracing.py` | Instrument every model call and tool call with a `Tracer`; save execution traces to JSON |
| `08_eval_harness.py` | Run a test suite against your agent; score with `ToolCallScorer` and `AnswerScorer` |
| `09_guardrails.py` | Three-layer defence: `InputGuardrail`, `ToolCallGuardrail`, `OutputGuardrail` |
| `10_resilience.py` | Four resilience patterns: `RetryPolicy`, `FallbackRegistry`, `CircuitBreaker`, `ToolTimeout` |

### Part 3 — AI-DLC Adaptive Workflows

| File | What you learn |
|------|----------------|
| `11_aidlc_inception.py` | Inception phase: `AuditLogger`, `StateTracker`, intent classification, clarifying questions, requirements generation, approval gates |
| `12_aidlc_design.py` | Application Design: question-driven design, `AnswerParser` quality checks, component + service + dependency documents |
| `13_aidlc_construction.py` | Construction: `CodeGenerationPlanner`, `CheckboxTracker` (immediate updates), `NFRAdvisor`, security extension opt-in |
| `14_aidlc_orchestrator.py` | Full adaptive workflow: `ExtensionRegistry`, `AdaptiveStageSelector` (ALWAYS/CONDITIONAL), brownfield detection, end-to-end run |

---

## Part 2 — Harness Engineering

A **harness** is infrastructure that wraps around an agent to control, observe,
test, and protect it — without touching the agent's core logic.

The four lessons in this section answer a single question:
> *How do you go from "it works in the demo" to "I can deploy this with confidence"?*

---

### Lesson 07 — Tracing & Observability

**Core idea:** You can't debug what you can't see.

When your agent gives a wrong answer in production, you need a time-ordered
record of every decision it made. That record is called an **execution trace**.

```
[  0ms]  MODEL  iter=1  185→42 tok   finish=tool_calls   84ms
[ 84ms]  TOOL   get_market_data({"ticker": "AAPL"})  →  '{"price": 185.5}'  1.2ms
[ 86ms]  MODEL  iter=2   227→31 tok  finish=stop         79ms
```

**What a trace captures per event:**

| Event type | Fields recorded |
|------------|-----------------|
| Model call | iteration, prompt tokens, completion tokens, latency, finish reason |
| Tool call  | tool name, arguments, result (truncated), latency |
| Run        | total wall time, model call count, tool call count, final answer |

**Key insight — where to instrument:**  
Tracing is *additive*. The agent loop is unchanged. You only add two lines
per event: one before and one after.

```python
t = tracer.start_model_call(iteration)          # ← before
response = client.chat.completions.create(...)
tracer.finish_model_call(t, response, iteration) # ← after
```

**What to do with traces:**
- Offline debugging: load `traces/trace_<id>.json`, find the iteration where things went wrong
- Regression detection: compare token counts and latency across versions
- Cost tracking: sum `prompt_tokens + completion_tokens` across all spans

---

### Lesson 08 — Eval Harness

**Core idea:** If you can't measure it, you can't improve it.

Every time you change your system prompt, add a tool, or switch models,
you need an automated way to answer: *did it get better or worse?*

**The structure of a test case:**

```python
EvalCase(
    id       = "affordability",
    question = "Can mayank afford 3 shares of MSFT?",
    expected_tools            = ["get_user_balance", "get_market_data"],
    expected_answer_contains  = ["mayank", "MSFT"],
)
```

- `expected_tools` — which tools the agent *must* call (order-independent, partial credit)
- `expected_answer_contains` — substrings the answer *must* include (case-insensitive, partial credit)

**The two scorers:**

| Scorer | What it measures | Partial credit |
|--------|------------------|----------------|
| `ToolCallScorer` | Did the agent call the right tools? | k/n expected tools called |
| `AnswerScorer` | Does the answer contain expected facts? | k/n substrings found |

**Writing good assertions:**
- Use substrings, not exact strings — LLMs paraphrase freely
- Prefer numeric values (they can't drift: `"185"` is better than `"Apple's price"`)
- `expected_tools` tests *behaviour*; `expected_answer_contains` tests *correctness*
- A case with no `expected_tools` still validates the answer (and vice versa)

**The eval loop:**

```
run eval suite → read report → fix the regression → run again
```

Treat eval accuracy as a metric you track across commits, just like test coverage.

---

### Lesson 09 — Guardrails

**Core idea:** The model is not the last line of defence — you are.

Guardrails are checks that run **around** the agent, not inside it.
The agent code is never modified; the `GuardedAgent` wrapper applies all three layers.

```
User Input
    │
  [ Layer 1: InputGuardrail  ]  ← block before the model ever sees it
    │
  [ Agent + tools            ]
        │
      [ Layer 2: ToolCallGuardrail ]  ← validate args before each tool runs
    │
  [ Layer 3: OutputGuardrail ]  ← check the answer before it reaches the user
    │
Final Response
```

**Layer 1 — InputGuardrail** checks:
- Length limit (refuse DoS-by-prompt attacks)
- Injection patterns: `"ignore all previous instructions"`, `"act as"`, `"you are now"`
- Harmful keywords: domain-specific blocklist

**Layer 2 — ToolCallGuardrail** checks per argument:
- Type (`str`, `int`, `float`)
- Max length — prevents the model passing a 50 KB string as a ticker symbol
- Allowed values — restrict to a known set (e.g. specific usernames)
- Numeric range — min/max bounds on numeric args

**Layer 3 — OutputGuardrail** checks:
- Empty or whitespace-only response
- Refusal phrases: `"I cannot"`, `"I am unable to"`, `"as an AI language model"`
- Disallowed content — domain-specific forbidden phrases in the answer

**When a guardrail fires:**  
A `GuardrailViolation` exception is raised with `layer` and `reason`.
The `GuardedAgent` catches it, logs it, and returns a safe fallback string.
The agent's loop is never reached (for input violations) or is stopped immediately.

---

### Lesson 10 — Resilience Engineering

**Core idea:** Build for failure — tools break, APIs timeout, models return garbage.

Four patterns compose into a `ResilientToolRunner`. The agent calls tools
exactly as before; resilience is completely transparent.

**Pattern 1 — RetryPolicy (exponential backoff + jitter)**

```
attempt 1 → fail → sleep 0.3s
attempt 2 → fail → sleep 1.1s
attempt 3 → fail → sleep 3.8s
attempt 4 → succeed ✓
```

Full jitter (`sleep = random.uniform(0, base * 2^attempt)`) is preferred
over fixed backoff because it prevents multiple clients retrying at the same
instant (the "thundering herd" problem).

**Pattern 2 — FallbackRegistry**

If the primary implementation fails (after retries), try a backup:

```
get_market_data → [csv_reader, mock_db]
                      ↓ fails        ↓ succeeds
```

Fallbacks are ordered (primary first). Register as many levels as you need.
The agent sees a valid result either way.

**Pattern 3 — CircuitBreaker**

```
CLOSED ──(N failures)──▶ OPEN ──(timeout elapsed)──▶ HALF-OPEN
   ▲                                                      │
   └──────────────(probe succeeds)───────────────────────┘
```

Why it matters: without a circuit breaker, retrying a broken tool wastes
tokens, time, and money. The circuit breaker *fails fast* — it rejects calls
immediately while the circuit is OPEN, without attempting the tool at all.

**Pattern 4 — ToolTimeout**

Enforces a maximum wall-clock time per tool call using a background thread.
If the tool exceeds the limit, `ToolTimeoutError` is raised and the retry
policy handles it — just like any other transient failure.

**Composition order:**

```
CircuitBreaker.call(
    RetryPolicy.execute(
        ToolTimeout.call(
            FallbackRegistry.call(primary_fn)
        )
    )
)
```

Each layer handles its own failure mode, in order from fastest to slowest:
circuit check (instant) → timeout (per attempt) → retry (across attempts) → fallback (last resort).

---

## Key Parameters for Gemma 4:E4B

```python
client.chat.completions.create(
    model="docker.io/ai/gemma4:E4B",
    messages=messages,
    temperature=1.0,          # Gemma 4 recommended — don't lower this much
    extra_body={
        "chat-template-kwargs": {"enable_thinking": True}  # activates <think> tokens
    }
)
```

`enable_thinking=True` lets the model reason internally before responding.
The `<think>...</think>` block shows up in `response.choices[0].message.content`.
You can strip it from memory to save tokens (see Lesson 1).

---

## Mental Model Checklist

### Before writing an agent (Part 1)

1. **What is the goal?** — one clear user intent per agent run
2. **What tools does it need?** — only give the model tools it actually needs
3. **When does it stop?** — define the termination condition clearly
4. **How does it remember?** — messages list (short) + file/DB (long)
5. **What happens when a tool fails?** — catch errors, inject them as observations

### Before deploying an agent (Part 2)

6. **Can I see what it did?** — add a `Tracer`; save execution traces to disk
7. **Do I have a test suite?** — write `EvalCase`s before you change the system prompt
8. **What can a hostile user do?** — add `InputGuardrail` + `ToolCallGuardrail`
9. **What if the answer is wrong?** — add `OutputGuardrail` to catch silent failures
10. **What if a tool goes down?** — add `RetryPolicy` + `FallbackRegistry` + `CircuitBreaker`

### Before running AI-DLC (Part 3)

11. **What is being built?** — run Inception; classify intent before any design
12. **Is the design question-driven?** — write `[Answer]:` questions; parse quality before generating components
13. **Is there a plan before code?** — create `code-generation-plan.md`; gate it; mark `[x]` immediately
14. **Which stages does this project actually need?** — let `AdaptiveStageSelector` decide; skip stages that add no value
15. **What opt-in rules apply?** — check `ExtensionRegistry`; deferred-load full rules only after opt-in

---

## Part 3 — AI-DLC Adaptive Workflows

**AI-DLC (AI-Driven Development Lifecycle)** is a model-agnostic methodology:
the rules live in structured system prompts, not inside any specific model.

Lessons 11–14 adapt AI-DLC for Gemma4, producing the same `aidlc-docs/`
artifact structure that any AI-DLC-compliant tool would produce.

> The core insight: **AI-DLC is model-agnostic by design.**
> Swapping Gemma4 for Claude or GPT-4o requires changing only the `client`
> and `MODEL` constant — all methodology logic is in the prompts.

---

### Lesson 11 — Inception Phase

**Core idea:** Determine WHAT and WHY before any HOW.

The biggest source of wasted development effort is building the wrong thing.
Inception gates this risk by requiring a formalised requirements document
before any design or code can begin.

```
User Request
    │
    ▼
IntentAnalyzer   ── classify: type / scope / complexity / depth_needed
    │
    ▼
ClarifyingQuestionsAgent  ── 5 targeted [Answer]: questions → written to disk
    │
    ▼
RequirementsAgent  ── reads answers → produces requirements.md
    │
    ▼
WorkflowPlanner    ── ALWAYS/CONDITIONAL logic → workflow-plan.md
    │
    ▼
⛔ APPROVAL GATE  ── human reviews requirements.md before any design
```

**Key classes:**

| Class | What it does |
|-------|-------------|
| `AuditLogger` | Appends every interaction to `audit.md` with ISO timestamps |
| `StateTracker` | Reads/writes `aidlc-state.md`; tracks stage status |
| `IntentAnalyzer` | Classifies request into structured JSON |
| `ClarifyingQuestionsAgent` | Writes `[Answer]:` tagged questions; reads answers back |
| `RequirementsAgent` | Produces numbered `requirements.md` |
| `WorkflowPlanner` | Decides which downstream stages to run |

**Artifacts produced:**
```
aidlc-docs/
  audit.md
  aidlc-state.md
  inception/requirements/requirement-verification-questions.md
  inception/requirements/requirements.md
  inception/workflow-plan.md
```

---

### Lesson 12 — Application Design

**Core idea:** Design is iterative and question-driven — not a single model call.

A single "design the system" prompt produces mediocre architecture.
AI-DLC splits design into two steps:
  1. Write questions → fill answers → check quality (Steps 8 + 9)
  2. Generate documents from validated answers

```
requirements.md
    │
    ▼
DesignQuestionWriter  ── writes application-design-plan.md (8 [Answer]: slots)
    │
    ▼
(human fills answers — or AUTO_APPROVE injects samples)
    │
    ▼
AnswerParser  ── parses answers; sends to Gemma4 for quality check
    │  └─ vague answer detected → follow-up appended to plan (Step 9)
    ▼
ComponentDesigner  ── components.md + component-methods.md
ServiceDesigner    ── services.md + component-dependency.md
    │
    ▼
⛔ APPROVAL GATE
```

**The vague-answer detector:**

When `AnswerParser.check_quality()` finds an answer shorter than 5 words,
or containing "TBD" / "maybe" / "various", it calls Gemma4 to generate a
targeted follow-up question and appends it to the plan file.
This prevents under-specified designs from reaching Construction.

**Artifacts produced:**
```
aidlc-docs/design/
  application-design-plan.md
  components.md
  component-methods.md
  services.md
  component-dependency.md
```

---

### Lesson 13 — Construction (Plan-Then-Generate)

**Core idea:** Never generate code without a plan. Checkboxes update immediately.

```
design documents
    │
    ▼
NFRAdvisor  ── assess performance + security risks
    │  └─ if security recommended → present SECURITY extension (opt-in)
    │
    ▼
CodeGenerationPlanner  ── writes code-generation-plan.md
    │                      "- [ ] Step 1: ..."  ...  "- [ ] Step N: ..."
    │
    ▼
⛔ APPROVAL GATE  ── plan must be approved before ANY code is generated
    │
    ▼
CodeGenerationExecutor  ── loop:
    │  1. determine target file from step description
    │  2. call Gemma4 to generate code (with optional security rules)
    │  3. write file to portfolio_calculator/
    │  4. immediately mark step [x] in plan  ← mandatory AI-DLC rule
    │
    ▼
All checkboxes [x] → construction complete
```

**Why immediate checkbox updates matter:**

If code generation crashes after step 3, re-running the lesson will skip
steps 1–3 (already `[x]`) and continue from step 4.  Without this, every
crash forces a full restart.

**Security extension rules (opt-in):**

| Rule | Description |
|------|-------------|
| `SEC-01` | Validate all external inputs before use |
| `SEC-02` | Never log raw user data; sanitise identifiers first |
| `SEC-03` | All errors caught; no raw stack traces to callers |

**Artifacts produced:**
```
aidlc-docs/construction/
  nfr-assessment.md
  code-generation-plan.md    ← checkboxes updated in real-time

portfolio_calculator/        ← generated code (workspace root, NOT aidlc-docs/)
  __init__.py
  data_reader.py
  calculator.py
  printer.py
  run_summary.py
  test_calculator.py
```

---

### Lesson 14 — Full Adaptive Workflow (Orchestrator)

**Core idea:** AI-DLC is adaptive — stages fire only when they add value.

```
User Request
    │
    ▼
detect_workspace()  ── greenfield vs brownfield?
    │                   brownfield → reuse existing requirements
    │
    ▼
ExtensionRegistry.present_and_select()
    │  ── summarise available extensions (before opt-in)
    │  ── deferred-load full rules only after opt-in
    │
    ▼
IntentAnalyzer  ── classify intent
    │
    ▼
AdaptiveStageSelector  ── ALWAYS/CONDITIONAL logic
    │  inception    → ALWAYS
    │  design       → complexity=medium|high OR scope!=single_module
    │  construction → ALWAYS
    │  build-test   → complexity=high OR scope=full_application
    │
    ▼
Print adaptive plan → ⛔ GATE → user approves
    │
    ▼
Execute each stage (or log "skipped") → ⛔ GATE between stages
    │
    ▼
Summary: stages run / skipped / artifacts produced
```

**Brownfield vs Greenfield:**

| Context | Behaviour |
|---------|-----------|
| Greenfield | Full Inception (generate requirements from scratch) |
| Brownfield | Reuse `aidlc-docs/inception/requirements/requirements.md` if it exists |

**Extension deferred loading:**

Extensions are shown as one-line summaries before opt-in.  Full rules are
only injected into code generation prompts after the user confirms.
This avoids polluting the system prompt with irrelevant rules.

**Demo scenarios:**

```bash
# Scenario B — full complex request (all stages run):
python3 lessons/14_aidlc_orchestrator.py

# Scenario A — simple request (adaptive; fewer stages):
python3 lessons/14_aidlc_orchestrator.py simple
```

**Artifacts produced:**
```
aidlc-docs/
  adaptive-workflow-plan.md   ← which stages run and why
  audit.md                    ← complete interaction log
  inception/                  ← requirements (if inception ran)
  design/                     ← design summary (if design ran)
  construction/               ← plan + NFR (if construction ran)
  build-and-test-instructions.md  ← (if build-test ran)

market_api/                   ← generated code (Scenario B)
```
