# Modern training migration candidate

Status: GPU acceptance pending, 2026-10-23. The dependency profile resolves, but production training readiness is not established.

The local training profile resolves VERL 0.9.1, vLLM 0.30.0, Transformers 5.10.4 and Torch 2.13.0 on x86-64 Python 3.11–3.12. Tinker is a separate environment using Cookbook 0.5.7 and Transformers 5.5.4. Do not install both profiles together. AutoGen runners also require a separate provider environment: AutoGen 0.7.5 requires protobuf 5, while vLLM 0.30 requires protobuf 6 through CUTLASS. Explicit conflicts prevent silently downgrading AutoGen to an obsolete package. Core Python now requires 3.11 or later so Pillow 13 can be used in every locked environment. The Tinker Transformers pin and isolated Chroma dependency still have unresolved advisories; this candidate does not claim a clean dependency audit.

The trainer uses VERL's unified actor/critic worker registration, fused reference routing, reward-loop manager, replica registry, LLM-server addresses and checkpoint-engine weight synchronization. Upstream conversion helpers retain padding, LoRA reference semantics and training metadata at TensorDict RPC boundaries. Rollout cleanup releases inference memory on errors.

CPU contract tests in `tests/instrumentation/test_verl_training_contract.py` use real VERL, Torch and TensorDict with fake remote worker responses. They do not initialize GPUs or a distributed worker cluster. Server-registration tests separately verify the instrumented replica contract. The Tinker tests cover its updated checkpoint and training-call signatures without a paid remote training call.

Before merge, validate the complete feature matrix:

- Full GRPO and GAE training, REMAX baseline generation, model rewards, KL/reference and LoRA modes, and dataset/daemon shapes.
- vLLM engine startup, instrumented token IDs, streaming and nonstreaming completions, checkpoint restore, weight updates, and distributed cleanup.
- Existing examples, image/multimodal inputs, FSDP and Megatron configurations, TRL/Unsloth compatibility, and supported CPU Python versions.
- Full formatting, type checks, documentation, hosted security analysis and required CI, plus a configured GPU training/inference runner.

No GPU runner is configured. The earlier legacy profile is not a safe fallback: its known advisories are the reason for this migration. Keep this branch in draft until runtime and compatibility validation is complete.

CI installs training, provider and Tinker environments separately. Full-project pyright uses a generated static-only `.venv-types` package path containing the training, provider and Tinker package directories in that order. This retains normal third-party stub resolution instead of treating installed dependencies as project sources. The generated directory has no interpreter and is never activated or used for runtime tests. Runtime tests use their own environments; the static search paths do not make the incompatible libraries a supported shared runtime.
