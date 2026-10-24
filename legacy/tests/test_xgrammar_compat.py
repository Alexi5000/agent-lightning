# Copyright (c) Microsoft. All rights reserved.

import subprocess
import sys

import pytest


def test_vllm_grammar_mask_and_rollback_contract() -> None:
    xgr = pytest.importorskip("xgrammar")
    torch = pytest.importorskip("torch")
    info = xgr.TokenizerInfo([chr(c) for c in range(32, 127)] + ["<eos>"], stop_token_ids=[95])
    compiler = xgr.GrammarCompiler(info, max_threads=1, cache_enabled=True, cache_limit_bytes=1024 * 1024)
    matcher = xgr.GrammarMatcher(compiler.compile_grammar('root ::= "ab"'), max_rollback_tokens=8)
    bitmask = xgr.allocate_token_bitmask(1, 96)
    matcher.fill_next_token_bitmask(bitmask, 0)
    logits = torch.zeros(1, 96)
    xgr.apply_token_bitmask_inplace(logits, bitmask)
    assert torch.where(torch.isfinite(logits))[1].tolist() == [ord("a") - 32]
    assert not matcher.accept_token(ord("z") - 32)
    assert matcher.accept_token(ord("a") - 32)
    assert matcher.accept_token(ord("b") - 32)
    matcher.rollback(1)
    assert matcher.accept_token(ord("b") - 32)
    assert matcher.accept_token(95)
    assert matcher.is_terminated()
    matcher.reset()
    assert not matcher.is_terminated()
    assert matcher.accept_token(ord("a") - 32)


@pytest.mark.parametrize("mode", ["json", "regex", "legacy-tags", "structured-tags"])
def test_vllm_supported_grammar_modes(mode: str) -> None:
    xgr = pytest.importorskip("xgrammar")
    info = xgr.TokenizerInfo([chr(c) for c in range(32, 127)] + ["<eos>"], stop_token_ids=[95])
    compiler = xgr.GrammarCompiler(info, max_threads=1)
    if mode == "json":
        compiled = compiler.compile_json_schema({"type": "object", "properties": {"ok": {"type": "boolean"}}})
        text = '{"ok":true}'
    elif mode == "regex":
        compiled = compiler.compile_regex("[ab]+")
        text = "abba"
    else:
        tags = [xgr.StructuralTagItem(begin="<tool>", schema={"type": "string"}, end="</tool>")]
        if mode == "legacy-tags":
            compiled = compiler.compile_structural_tag(tags, ["<tool>"])
            xgr.Grammar.from_structural_tag(tags, ["<tool>"])
        else:
            tag = xgr.StructuralTag.from_legacy_structural_tag(tags, ["<tool>"])
            compiled = compiler.compile_structural_tag(tag)
            xgr.Grammar.from_structural_tag(tag)
        text = '<tool>"ok"</tool>'
    matcher = xgr.GrammarMatcher(compiled)
    for character in text:
        assert matcher.accept_token(ord(character) - 32)
    assert matcher.accept_token(95)
    assert matcher.is_terminated()


def test_deeply_nested_untrusted_grammar_rejected_without_crash() -> None:
    pytest.importorskip("xgrammar")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import xgrammar as xgr; xgr.Grammar.from_ebnf('root ::= ' + '(' * 30000 + 'a')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    assert "RuntimeError" in result.stderr
    assert "nest" in result.stderr.lower()


def test_vllm_huggingface_tokenizer_conversion() -> None:
    xgr = pytest.importorskip("xgrammar")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"<unk>": 0, "a": 1, "b": 2, "<eos>": 3}, unk_token="<unk>")
    )
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer))
    compiler = xgr.GrammarCompiler(info, max_threads=1)
    matcher = xgr.GrammarMatcher(compiler.compile_grammar('root ::= "ab"'))
    bitmask = xgr.allocate_token_bitmask(1, len(tokenizer))
    matcher.fill_next_token_bitmask(bitmask, 0)
    logits = torch.zeros(1, len(tokenizer))
    xgr.apply_token_bitmask_inplace(logits, bitmask)
    assert torch.where(torch.isfinite(logits))[1].tolist() == [1]
    assert matcher.accept_token(1)
    assert matcher.accept_token(2)
    assert matcher.accept_token(tokenizer.eos_token_id)
    assert matcher.is_terminated()
