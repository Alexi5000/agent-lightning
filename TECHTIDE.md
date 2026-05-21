# TechTide AI — Why This Fork Exists

## The Problem

Training RL agents at scale requires infrastructure most teams build from scratch. When you're running PPO training loops with custom reward functions across multiple environments, you need a framework that handles the distributed coordination, trace collection, and model serving — not just the algorithm.

## Why Agent Lightning

Microsoft's Agent Lightning gives us a production-ready RL training pipeline that works with any agent framework. The verl integration handles distributed PPO training, the trace system captures agent behavior for reward computation, and the LLM proxy manages model serving during rollouts.

We use this internally at TechTide to train and evaluate agents for client deployments. When a client needs an agent that improves over time from real interaction data, Agent Lightning is how we close the loop from traces → rewards → policy updates.

## What TechTide Uses This For

- **Agent evaluation harnesses** — Structured evaluation of agent behavior before production deployment
- **Custom reward pipelines** — Computing rewards from multi-turn agent traces for domain-specific tasks
- **Training infrastructure** — Distributed PPO training across GPU clusters for agent fine-tuning
- **Benchmark automation** — Automated agent benchmarking against standardized task suites

## Upstream Contributions

We contribute bug fixes and code quality improvements back to the upstream project:

| PR | Description |
|----|-------------|
| [#528](https://github.com/microsoft/agent-lightning/pull/528) | Fix missing `gts` argument in `_dump_generations` call (issue #492) |
| [#529](https://github.com/microsoft/agent-lightning/pull/529) | Fix pyright type errors in weave instrumentation (issue #496) |
| [#530](https://github.com/microsoft/agent-lightning/pull/530) | Replace bare `print()` calls with structured logging in verl daemon |

## Architecture Notes

Agent Lightning is a Python framework with:
- **Core library** (`agentlightning/`): Adapters, algorithms (APO, verl/PPO), tracing, LLM proxy, and store
- **Dashboard** (`dashboard/`): Web UI for training monitoring
- **Examples** (`examples/`): 10+ reference implementations (calc-x, SQL, RAG, Claude Code, etc.)
- **Contrib** (`contrib/`): Community recipes and extensions

Key strengths:
- **Framework-agnostic**: Works with LangChain, CrewAI, custom agents via adapter pattern
- **Distributed training**: verl-based PPO with Ray workers and multi-GPU support
- **Comprehensive tracing**: AgentOps, vLLM, LiteLLM, and Weave instrumentation
- **73 test files** with strong modular coverage

---

*This fork is maintained by [TechTide AI](https://github.com/TechTideOhio) as part of our agent training infrastructure stack.*
