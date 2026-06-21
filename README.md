# 🧪 Evals Framework for LLM Agentic Applications

A generic, extensible evaluation framework for testing LLM-powered agentic applications. Built in two phases:

1. **Phase 1** — An example multi-tool research assistant agent (the system-under-test)
2. **Phase 2** — A decoupled evals framework that can evaluate *any* agent via a thin adapter interface

> **Status**: Both Phase 1 (Example Agent) and Phase 2 (Evals Framework) are fully complete!

---

## 📖 Table of Contents

* [🏗 Architecture](#-architecture)
* [🚀 Quick Start](#-quick-start)
  * [1. Install Dependencies](#1-install-dependencies)
  * [2. Configure API Keys](#2-configure-api-keys)
  * [3. Run the Example Agent Interactively](#3-run-the-example-agent-interactively)
  * [4. Run the Evaluation Suite](#4-run-the-evaluation-suite)
* [🛠 Extending the Framework](#-extending-the-framework)
  * [How to Add a New Agent](#how-to-add-a-new-agent)
  * [How to Add New Eval Cases](#how-to-add-new-eval-cases)
  * [How to Add New Scorers](#how-to-add-new-scorers)
  * [How to Run Evals in CI](#how-to-run-evals-in-ci)
* [🎯 Project Roadmap](#-project-roadmap)
  * [✅ Phase 1: Example Agent](#-phase-1-example-agent)
  * [✅ Phase 2: Generic Evals Framework](#-phase-2-generic-evals-framework)
* [License](#license)

---

## 🏗 Architecture

```
evals-framework/
├── agent/                        # Example agentic application
│   ├── app.py                    # Main agent orchestration loop + AgentTrace
│   ├── cli.py                    # Interactive REPL for manual testing
│   └── tools/                    # Tool implementations (search, calculator, etc.)
├── evals/                        # Generic evals framework
│   ├── adapters/                 # Bridges to different agents (e.g. ExampleAgentAdapter)
│   ├── configs/                  # Evaluation run configuration yaml files
│   ├── core/                     # Core execution logic (Runner, Reporter, Datasets)
│   ├── datasets/                 # Evaluation test cases (JSONL files)
│   ├── scorers/                  # Evaluation criteria (Deterministic and LLM Judge)
│   ├── cli.py                    # CLI tool for executing evals
│   └── conftest.py               # Pytest plugin for running evals in CI/CD
├── tests/                        # Framework unit tests
├── pyproject.toml
└── .env.example
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone <repo-url> evals-framework
cd evals-framework
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configure API Keys

```bash
cp .env.example .env
```
Edit `.env` to add your OpenAI, Anthropic, or Gemini API keys.

### 3. Run the Example Agent Interactively

Verify your agent works locally:
```bash
evals-agent --verbose
```

### 4. Run the Evaluation Suite

Run the full evaluation suite against the example agent and output to the terminal:
```bash
evals-run run --config evals/configs/default.yaml --format all
```

Compare two eval runs:
```bash
evals-run compare --baseline evals/results/run1.json --current evals/results/run2.json
```

Check the dataset statistics:
```bash
evals-run dataset-info --path evals/datasets
```

---

## 🛠 Extending the Framework

The evals framework is built to be completely generic. You can use it to test **your own** AI agents, add custom test cases, or define custom scoring rules.

### How to Add a New Agent

To test your own agent, create a new adapter that implements the `AgentAdapter` interface. This translates your agent's custom output format into the framework's standard `AgentOutput` format.

```python
from evals.core.interfaces import AgentAdapter, AgentOutput

class MyCustomAgentAdapter(AgentAdapter):
    def __init__(self, my_agent):
        self.agent = my_agent

    async def execute(self, input: str) -> AgentOutput:
        # Run your agent
        raw_result = await self.agent.run(input)
        
        # Translate to framework format
        return AgentOutput(
            input=input,
            output=raw_result.text,
            steps=[...], # convert trace steps
            total_steps=raw_result.step_count,
        )

    def reset(self) -> None:
        self.agent.clear_memory()

    def get_info(self) -> dict:
        return {"name": "MyCustomAgent", "version": "1.0"}
```

### How to Add New Eval Cases

Eval cases are defined in `.jsonl` files in the `evals/datasets/` directory. Simply add a new line to any JSONL file (or create a new one):

```json
{
  "id": "new-test-case-1",
  "input": "Calculate 25 * 4",
  "expected_tool_calls": [{"tool_name": "calculator"}],
  "expected_outcome": "100",
  "tags": ["math", "simple"],
  "difficulty": "easy"
}
```

### How to Add New Scorers

You can create custom rule-based heuristics to grade agent outputs by extending `BaseScorer`.

```python
from evals.scorers.base import BaseScorer
from evals.core.interfaces import ScoreResult

class WordCountScorer(BaseScorer):
    @property
    def name(self) -> str: return "word_count"
    
    @property
    def threshold(self) -> float: return 1.0

    async def score(self, case, output) -> ScoreResult:
        words = len(output.output.split())
        passed = words >= 10
        return ScoreResult(
            scorer_name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            threshold=self.threshold,
            reasoning=f"Found {words} words."
        )
```

Then add your scorer to the `CompositeScorer` pipeline.

### How to Run Evals in CI

The framework natively integrates with Pytest. You can run subsets of your evaluations quickly in your CI pipelines using pytest markers and keyword filtering!

For example, to run only the "regression" cases in CI:
```bash
pytest evals/ -v -k "regression"
```

Because `evals/conftest.py` automatically parametrizes the dataset into native Pytest tests, your CI pipeline will correctly fail the build if the agent regresses on golden test cases.

---

## 🎯 Project Roadmap

### ✅ Phase 1: Example Agent
- [x] Abstract base tool interface (`BaseTool`, `ToolCall`, `ToolResult`)
- [x] Four simulated tools (web search, calculator, weather, knowledge base)
- [x] Conversation memory with LLM-format export
- [x] LLM-powered task planner (OpenAI / Anthropic / Gemini)
- [x] Rule-based safety filter (injection detection, harmful content blocking)
- [x] Agent orchestration with full `AgentTrace` output
- [x] Interactive CLI with Rich formatting

### ✅ Phase 2: Generic Evals Framework
- [x] Abstract interfaces (`AgentAdapter`, `AgentOutput`, `EvalCase`, `ScoreResult`)
- [x] Deterministic scorers (tool selection, argument matching, trajectory efficiency, safety, exact match)
- [x] Cost and Latency Budgeting scorer (`CostLatencyScorer`)
- [x] LLM-as-judge scorer with bias mitigation
- [x] Groundedness LLM Scorer for strict hallucination detection
- [x] Composite scorer with configurable weights
- [x] Eval dataset loader (JSONL) with filtering and sampling
- [x] Multi-turn conversation execution and trace aggregation
- [x] Async eval runner with concurrency control and timeouts
- [x] Reporter (terminal, JSON, markdown) with run-to-run comparison
- [x] CLI for running and comparing eval runs
- [x] Pytest integration for CI/CD
- [x] Framework unit tests

See [`developer_walkthrough.md`](./developer_walkthrough.md) for the full design specifications.

---

## License

MIT
