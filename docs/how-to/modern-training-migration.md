# Modern training migration candidate

Status: draft, 2026-10-19. This branch is not ready for release or merge.

The local training profile resolves VERL 0.9.1, vLLM 0.26.0, Transformers 5.10.0 and Torch 2.11.0 on x86-64 Python 3.11–3.12. Tinker is a separate environment using Cookbook 0.5.7 and Transformers 5.5.4. Do not install both profiles together. Core Python support is unchanged. The Tinker Transformers pin and isolated Chroma/NLTK dependencies still have unresolved advisories; this candidate does not claim a clean dependency audit.

The trainer uses VERL's unified actor/critic worker registration, fused reference routing, reward-loop manager, replica registry, LLM-server addresses and checkpoint-engine weight synchronization. Upstream conversion helpers retain padding, LoRA reference semantics and training metadata at TensorDict RPC boundaries. Rollout cleanup releases inference memory on errors.

CPU contract tests in `tests/instrumentation/test_verl_training_contract.py` use real VERL, Torch and TensorDict with fake remote worker responses. They do not initialize GPUs or a distributed worker cluster. Server-registration tests separately verify the instrumented replica contract. The Tinker tests cover its updated checkpoint and training-call signatures without a paid remote training call.

Before merge, validate the complete feature matrix:

- Full GRPO and GAE training, REMAX baseline generation, model rewards, KL/reference and LoRA modes, and dataset/daemon shapes.
- vLLM engine startup, instrumented token IDs, streaming and nonstreaming completions, checkpoint restore, weight updates, and distributed cleanup.
- Existing examples, image/multimodal inputs, FSDP and Megatron configurations, TRL/Unsloth compatibility, and supported CPU Python versions.
- Full formatting, type checks, documentation, hosted security analysis and required CI, plus a configured GPU training/inference runner.

No GPU runner is configured. The earlier legacy profile is not a safe fallback: its known advisories are the reason for this migration. Keep this branch in draft until runtime and compatibility validation is complete.
