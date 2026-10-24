# Copyright (c) Microsoft. All rights reserved.

import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "tool_input, expected",
    [
        ('{query: "local evidence", count: 2,}', {"query": "local evidence", "count": 2}),
        ('{"query": "local evidence", "enabled": true}', {"query": "local evidence", "enabled": True}),
        ('[{"query": "one"}, {"query": "two"}]', [{"query": "one"}, {"query": "two"}]),
    ],
)
def test_crewai_repairs_tool_arguments(tool_input: str, expected: object) -> None:
    parser = pytest.importorskip("crewai.agents.parser")
    result = parser.parse(f"Thought: Inspect local evidence.\nAction: Search\nAction Input: {tool_input}")
    assert isinstance(result, parser.AgentAction)
    assert result.tool == "Search"
    assert json.loads(result.tool_input) == expected


def test_crewai_preserves_final_answer() -> None:
    parser = pytest.importorskip("crewai.agents.parser")
    result = parser.parse("Thought: Evidence checked.\nFinal Answer: No provider call required.")
    assert isinstance(result, parser.AgentFinish)
    assert result.output == "No provider call required."


def test_circular_schema_fails_without_hanging() -> None:
    pytest.importorskip("json_repair")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from json_repair import loads; "
            "loads('{}', schema={'$ref': '#/definitions/a', "
            "'definitions': {'a': {'$ref': '#/definitions/a'}}})",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0
    assert "SchemaDefinitionError" in result.stderr
    assert "Circular $ref" in result.stderr
