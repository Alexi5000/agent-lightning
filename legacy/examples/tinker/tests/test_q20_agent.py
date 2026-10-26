# Copyright (c) Microsoft. All rights reserved.

"""Offline contract tests for the framework-free 20 Questions flow."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest
from q20_agent import ChatModel, SearchTool, TwentyQuestionsFlow


class FakeEndpoint:
    """Scripted OpenAI-compatible chat-completions endpoint."""

    def __init__(self, replies: List[Dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.requests: List[Dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        self.requests.append(body)
        assert self.replies, "unexpected extra request"
        message = {"role": "assistant", **self.replies.pop(0)}
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{len(self.requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            },
        )

    def model(self, name: str, **kwargs: Any) -> ChatModel:
        return ChatModel(
            model=name,
            base_url="http://fake.local/v1",
            api_key="test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)),
            **kwargs,
        )


def answer(yes_or_no: str, correct: bool) -> Dict[str, Any]:
    return {"content": json.dumps({"brief_reason": None, "yes_or_no": yes_or_no, "correct": correct})}


def search_call(call_id: str, query: str) -> Dict[str, Any]:
    return {
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "search", "arguments": json.dumps({"search_query": query})},
            }
        ],
    }


@pytest.mark.asyncio
async def test_correct_guess_ends_game_with_structured_answer() -> None:
    player = FakeEndpoint([{"content": "  Is it a cat?  "}])
    answerer = FakeEndpoint([answer("yes", True)])
    flow = TwentyQuestionsFlow(
        player_llm=player.model("player-model"),
        answer_llm=answerer.model("answer-model", reasoning_effort="low"),
    )

    state = await flow.kickoff_async({"answer": "cat", "category": "animal"})

    assert state.correct is True
    assert state.turn_index == 1
    assert [(t.question, t.response) for t in state.interactions] == [("Is it a cat?", "yes")]
    player_request = player.requests[0]
    assert player_request["model"] == "player-model"
    assert "tools" not in player_request
    assert "animal" in player_request["messages"][-1]["content"]
    answer_request = answerer.requests[0]
    assert answer_request["model"] == "answer-model"
    assert answer_request["reasoning_effort"] == "low"
    assert answer_request["response_format"]["type"] == "json_schema"
    assert answer_request["response_format"]["json_schema"]["name"] == "AnswererResponse"
    assert '"cat"' in answer_request["messages"][-1]["content"]
    assert "Is it a cat?" in answer_request["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_player_search_tool_round_trip() -> None:
    player = FakeEndpoint([search_call("call-1", "  largest land animal  "), {"content": "Is it an elephant?"}])
    search = FakeEndpoint([{"content": " The African elephant is the largest land animal. "}])
    answerer = FakeEndpoint([answer("yes", True)])
    tool = SearchTool(model=search.model("search-model", reasoning_effort="none"))
    flow = TwentyQuestionsFlow(
        player_llm=player.model("player-model"), answer_llm=answerer.model("answer-model"), search_tool=tool
    )

    state = await flow.kickoff_async({"answer": "elephant", "category": "animal"})

    assert state.correct is True
    assert state.num_tool_calls == 1
    assert tool.num_called == 1
    assert player.requests[0]["tools"][0]["function"]["name"] == "search"
    assert "search_query" in player.requests[0]["tools"][0]["function"]["parameters"]["properties"]
    followup = player.requests[1]["messages"]
    assert followup[-2]["role"] == "assistant"
    assert followup[-2]["tool_calls"][0]["id"] == "call-1"
    assert followup[-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "The African elephant is the largest land animal.",
    }
    assert "largest land animal" in search.requests[0]["messages"][-1]["content"]
    assert search.requests[0]["reasoning_effort"] == "none"
    assert state.interactions[0].question == "Is it an elephant?"


@pytest.mark.asyncio
async def test_player_tool_rounds_are_bounded() -> None:
    # Two tool rounds are allowed; the final call is made without tools to force a question.
    player = FakeEndpoint([search_call("a", "x"), search_call("b", "y"), {"content": "Is it a dog?"}])
    search = FakeEndpoint([{"content": "fact one"}, {"content": "fact two"}])
    answerer = FakeEndpoint([answer("yes", True)])
    flow = TwentyQuestionsFlow(
        player_llm=player.model("p"),
        answer_llm=answerer.model("a"),
        search_tool=SearchTool(model=search.model("s")),
    )

    state = await flow.kickoff_async({"answer": "dog", "category": "animal"})

    assert [("tools" in r) for r in player.requests] == [True, True, False]
    assert state.num_tool_calls == 2
    assert state.interactions[0].question == "Is it a dog?"


@pytest.mark.asyncio
async def test_search_tool_rejects_bad_arguments_and_truncates_queries() -> None:
    search = FakeEndpoint([{"content": "summary"}])
    tool = SearchTool(model=search.model("s"))

    assert (await tool.run_tool_call("{}")).startswith("Invalid search arguments")
    assert await tool.run("   ") == "No query provided."
    assert await tool.run_tool_call(json.dumps({"search_query": "q" * 600})) == "summary"
    sent = search.requests[0]["messages"][-1]["content"]
    assert "q" * 500 + "..." in sent
    assert "q" * 501 not in sent
    assert tool.num_called == 2


@pytest.mark.asyncio
async def test_malformed_and_fenced_answers() -> None:
    player = FakeEndpoint([{"content": "Is it alive?"}, {"content": "Is it a cat?"}])
    answerer = FakeEndpoint(
        [
            {"content": "I cannot answer in JSON"},
            {"content": '```json\n{"brief_reason": "match", "yes_or_no": "yes", "correct": true}\n```'},
        ]
    )
    flow = TwentyQuestionsFlow(player_llm=player.model("p"), answer_llm=answerer.model("a"))

    state = await flow.kickoff_async({"answer": "cat", "category": "animal"})

    assert [(t.question, t.response) for t in state.interactions] == [
        ("Is it alive?", "n/a"),
        ("Is it a cat?", "yes"),
    ]
    assert state.correct is True
    assert state.turn_index == 2
    assert "Question #1: Is it alive?\nResponse #1: n/a" in player.requests[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_game_is_lost_after_twenty_turns() -> None:
    player = FakeEndpoint([{"content": f"Question {i}?"} for i in range(20)])
    answerer = FakeEndpoint([answer("no", False) for _ in range(20)])
    flow = TwentyQuestionsFlow(player_llm=player.model("p"), answer_llm=answerer.model("a"))

    state = await flow.kickoff_async({"answer": "cat", "category": "animal"})

    assert state.correct is False
    assert state.turn_index == 20
    assert len(state.interactions) == 20
    assert "You have 1 turns left" in player.requests[-1]["messages"][-1]["content"]
    assert state.model_dump()["answer"] == "cat"
