# Copyright (c) Microsoft. All rights reserved.

"""Framework-free 20 Questions agents tailored for Agent-lightning demos.

This module wires up the player, answerer, and auxiliary tooling that power the
20 Questions examples under ``examples/tinker``.

It mirrors the high-level game loop used in the original Tinker Cookbook example,
with multiple interacting agents and a simulated web search tool exposed through
OpenAI-compatible tool calling. Earlier revisions used CrewAI, whose required
ChromaDB dependency carries unpatched critical advisories; the same player,
answerer and search behavior now runs directly on the OpenAI-compatible
chat-completions API, so every call still flows through the Agent-lightning
LLM proxy and tracer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, cast

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console

console = Console()


@dataclass
class ChatModel:
    """Connection settings for one OpenAI-compatible chat model.

    Attributes:
        model: Model name as understood by the endpoint (no provider prefix).
        base_url: OpenAI-compatible base URL. ``None`` uses the OpenAI SDK default.
        api_key: API key for the endpoint.
        timeout: Request timeout in seconds.
        reasoning_effort: Optional reasoning effort forwarded to the endpoint.
        http_client: Optional HTTP client, mainly for tests and custom transports.
    """

    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 120.0
    reasoning_effort: Optional[str] = None
    http_client: Optional[httpx.AsyncClient] = None
    _client: Optional[AsyncOpenAI] = field(default=None, init=False, repr=False)

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "dummy",
                timeout=self.timeout,
                http_client=self.http_client,
            )
        return self._client

    async def chat(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """Run one chat completion and return the first choice message as a plain dict."""
        if self.reasoning_effort is not None:
            kwargs.setdefault("reasoning_effort", self.reasoning_effort)
        create: Any = self.client.chat.completions.create
        completion = cast(ChatCompletion, await create(model=self.model, messages=messages, **kwargs))
        return completion.choices[0].message.model_dump(exclude_none=True)


class AnswererResponse(BaseModel):
    """Response schema for the answerer in 20 Questions game."""

    # Keep this short; do NOT ask for chain-of-thought
    brief_reason: Optional[str] = Field(description="1-2 sentences justification (optional, high level only).")
    yes_or_no: Literal["yes", "no", "n/a"] = Field(
        description="Whether the correct answer to the player's question is yes, no, or not applicable."
    )
    correct: bool = Field(
        description="Whether the player has correctly guessed the entity, and the game should end now."
    )


PLAYER_QUERY_TEMPLATE = """You are playing 20 Questions as the **Player**.
Ask one high-information **yes/no** question that most reduces the remaining possibility space.
If you think you have figured out the secret entity, ask a direct guess in the form: **"Is it <entity>?"**

THIS IS TURN #{turn_index} OF 20. You have {remaining_turns} turns left. The quicker you make a correct guess, the higher your score.

## Important assumptions
- Your answer BELONGS TO the category of "{category}".
- The answer is **straightforward, familiar, and commonly known**. They can be at most 3 words long (and only one word long in a majority of cases).
- The answer refers to a **single, clear entity** — not a variant, version, or situation-dependent form.

## What you have: Game history (Q/A pairs):

{history}

## Strategy guidelines (concise, practical)
- **Start broad, then narrow**: prioritize sub-category-level splits first (within {category}), then mid-level properties, then identifiers.
- **Binary partitioning**: prefer questions that split the remaining set near the middle.
- **Property over identity**: ask about features/roles/usages before naming brands, species, or specific titles.
- **Contradiction guard**: if past answers imply a contradiction, ask a short reset/sanity-check question to reconcile.
- **n/a handling**: if the last answer was **n/a**, pivot wording to a clearer, factual property.
- **Endgame**: if you have only one turn left, make a direct guess.

## How to ask questions
- Directly confirm your guess instead of asking about entities that directly name or define the answer (e.g., "Is it a type of pizza?" if "Pizza" is an option).
- Avoid questions that depend on subjective or situational conditions (e.g., "Would most people consider it artistic?").
- You are encouraged to use the search tool to check factual details or implications behind your potential question. This helps ensure your reasoning is grounded, accurate, and avoids irrelevant or trivial inquiries. However, you can only use the search tool at most once for each question; you must not use the search tool consecutively without asking a question in between.

## Output format (critical)

- Output **only** one yes/no question on a single line.
- No preamble, no numbering, no quotes, no meta commentary.
- Keep it concise, under 50 words.
- If guessing: use the form **Is it <entity>?**

Now produce your single best next question."""


ANSWERER_QUERY_TEMPLATE = """You are the **Answerer** in 20 Questions. Answer yes/no questions truthfully about the secret entity; mark correct if guessed exactly.

Your secret entity is: "{answer}". It belongs to the category of "{category}".

## The player's current question

{next_question}

## Rules

- Respond only with a structured yes/no evaluation about the entity.
- Be concise, objective, and consistent with previous answers.
- Never reveal the entity unless the player guessed correctly.
- If you don't know the answer, for example, the information is never publicly known, or the question is irrelevant to the entity's nature, answer **"n/a"**.

### Primary-sense rule (important)
- Answer based on the entity's **primary, literal identity**, not metaphorical associations or what it “can represent.”
  (Example: a famous building is **not** a "symbol" just because people call it a symbol of love.)
- Use the multi-meaning rule **only** when the entity's **name itself** has multiple mainstream senses (e.g., “football” the sport vs. the ball). Otherwise, stick to the primary sense.

### Handling unknown or irrelevant questions
- If the question asks about something **not publicly known**, **not factual**, **ambiguous**, or **irrelevant**, respond **"n/a"**.
- Use **"n/a"** only when a yes/no would be **misleading or nonsensical**.
- Examples:
  - "Does it have parents?" -> *n/a* (not meaningful for a place or object)
  - "Is it alive?" -> valid for all entities (answer yes/no if possible)
  - "Is it an animal?" -> *n/a* if the entity is a person, as this can be ambiguous.
  - "Does it post on social media?" -> *n/a* unless the entity is a living or fictional character known for doing so.
  - "Is the chair branded by a famous manufacturer?" -> *n/a* for a general object like "chair".

### Handling ambiguous entities
If the secret entity truly has multiple common meanings:
- Answer **"yes"** if the question is true for **any** major, well-recognized meaning.
- Answer **"no"** only if it's false for **all** reasonable interpretations.
- Do **not** stretch to metaphors or loose associations.

### Handling direct guesses
If the player's question is a direct guess ("Is it ...?"):
- Set **correct = true** if the guess is a close match in meaning to the secret entity (e.g., “Is it cell phone?” ≈ “Smartphone”).
- Otherwise, set **correct = false**.
"""


SEARCH_PROMPT_TEMPLATE = """You are simulating a web search.

Query: "{search_query}"

Return a concise, factual summary (2-4 sentences) of the most relevant information you would find online.
Avoid speculation, filler, or references to being an AI. Just give the facts."""


SEARCH_SYSTEM_PROMPT = (
    "You simulate a web search engine, producing factual, neutral summaries. "
    "Given a user's search query, return a concise, factual summary as if retrieved from reliable sources. "
    "Do not fabricate sources or URLs. Focus on core, verifiable facts. "
    "Never disclose that you are a simulator of a search engine."
)

MAX_SEARCH_QUERY_CHARS = 500


class SearchToolInput(BaseModel):
    """Schema for search tool input."""

    search_query: str = Field(
        ...,
        description="A short, factual query describing what to search for (e.g., 'capital of France', 'biography of Ada Lovelace').",
    )


class SearchTool:
    """A mock web search tool powered by an LLM.

    This class mimics a real search engine call by using a lightweight LLM model.
    It can later be replaced by a real API (like Serper or Bing) without changing its interface.
    """

    name: str = "search"
    description: str = "Search the web. Provide a concise, factual summary of what is known about the given topic."

    def __init__(self, model: ChatModel) -> None:
        self.model = model
        self.num_called = 0

    def tool_spec(self) -> Dict[str, Any]:
        """OpenAI function-calling specification for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": SearchToolInput.model_json_schema(),
            },
        }

    async def run(self, search_query: str) -> str:
        """Perform a mocked search request using an LLM."""
        self.num_called += 1
        # Safety: ensure input is not too long or empty
        search_query = search_query.strip()
        if not search_query:
            return "No query provided."
        if len(search_query) > MAX_SEARCH_QUERY_CHARS:
            search_query = search_query[:MAX_SEARCH_QUERY_CHARS] + "..."

        prompt = SEARCH_PROMPT_TEMPLATE.format(search_query=search_query)
        message = await self.model.chat(
            [{"role": "system", "content": SEARCH_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        )
        return str(message.get("content") or "").strip()

    async def run_tool_call(self, arguments: str) -> str:
        """Execute a tool call whose arguments are a JSON object matching :class:`SearchToolInput`."""
        try:
            parsed = SearchToolInput.model_validate_json(arguments or "{}")
        except ValidationError as exc:
            return f"Invalid search arguments: {exc.errors(include_url=False, include_input=False)}"
        return await self.run(parsed.search_query)


class Turn(BaseModel):
    """Represents a single turn in a 20 Questions game.

    Attributes:
        question: The question asked by the player.
        response: The answerer's "yes" or "no" or "n/a" response.
    """

    question: str
    response: Literal["yes", "no", "n/a"]


class TwentyQuestionsGameState(BaseModel):
    """State of a 20 Questions game session.

    Attributes:
        answer: The secret entity the player is trying to guess.
        category: The category of the secret entity.
        correct: Whether the player has guessed correctly.
        num_tool_calls: Number of search tool calls made during the game.
        next_question: The current question being processed.
        turn_index: Current turn number (1-20).
        interactions: History of question-answer turns.
    """

    answer: str = ""
    category: str = ""
    correct: bool = False
    num_tool_calls: int = 0
    next_question: str = ""
    turn_index: int = 1
    interactions: List[Turn] = Field(default_factory=list)  # type: ignore

    def render_history(self) -> str:
        """Render the game history as a formatted string.

        Returns:
            Formatted string showing all questions and responses.
        """
        return "\n\n".join(
            [
                f"Question #{i}: {turn.question}\nResponse #{i}: {turn.response}"
                for i, turn in enumerate(self.interactions, start=1)
            ]
        )


PLAYER_SYSTEM_PROMPT = (
    "You are a player in a game of 20 questions. "
    "Goal: minimize uncertainty and identify the hidden entity within 20 yes/no questions. "
    "You are a focused reasoner who uses binary-partition questions to narrow down the remaining possibilities."
)

ANSWERER_SYSTEM_PROMPT = "Reply only with a JSON object matching the requested schema."

# Maximum model calls per player turn when the search tool is available. The player is
# instructed to search at most once per question; extra rounds absorb malformed tool use.
PLAYER_MAX_ITER = 3


class TwentyQuestionsFlow:
    """Run one 20 Questions game between a player and an answerer.

    The player may call the optional search tool through OpenAI-compatible tool calling.
    The answerer returns a structured :class:`AnswererResponse`.
    """

    def __init__(
        self,
        *,
        player_llm: ChatModel,
        answer_llm: ChatModel,
        search_tool: Optional[SearchTool] = None,
    ) -> None:
        self.player_llm = player_llm
        self.answer_llm = answer_llm
        self.search_tool = search_tool
        self.state = TwentyQuestionsGameState()

    async def kickoff_async(self, inputs: Dict[str, Any]) -> TwentyQuestionsGameState:
        """Play a full game. ``inputs`` must provide ``answer`` and ``category``."""
        self.state = TwentyQuestionsGameState(
            answer=str(inputs["answer"]),
            category=str(inputs["category"]),
        )
        while True:
            await self.ask_question()
            await self.answer_question()
            if self.game_should_continue() == "game_over":
                break
        self.finish()
        return self.state

    async def ask_question(self) -> None:
        """Generate the next question from the player agent."""
        query = PLAYER_QUERY_TEMPLATE.format(
            history=self.state.render_history(),
            turn_index=self.state.turn_index,
            remaining_turns=20 - self.state.turn_index + 1,
            category=self.state.category,
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": PLAYER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        question = ""
        for iteration in range(PLAYER_MAX_ITER):
            use_tools = self.search_tool is not None and iteration < PLAYER_MAX_ITER - 1
            kwargs: Dict[str, Any] = {}
            if use_tools and self.search_tool is not None:
                kwargs["tools"] = [self.search_tool.tool_spec()]
            message = await self.player_llm.chat(messages, **kwargs)
            tool_calls: List[Dict[str, Any]] = list(message.get("tool_calls") or [])
            if not use_tools or not tool_calls or self.search_tool is None:
                question = str(message.get("content") or "").strip()
                break
            messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": tool_calls})
            for tool_call in tool_calls:
                function: Dict[str, Any] = tool_call.get("function") or {}
                if function.get("name") == self.search_tool.name:
                    result = await self.search_tool.run_tool_call(str(function.get("arguments") or ""))
                else:
                    result = f"Unknown tool: {function.get('name')!r}"
                messages.append({"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": result})
        console.print(f"[bold red]Player (Turn {self.state.turn_index}):[/bold red] {question}")
        if self.search_tool is not None:
            self.state.num_tool_calls = self.search_tool.num_called
        self.state.next_question = question

    async def _call_answerer(self, query: str) -> AnswererResponse:
        message = await self.answer_llm.chat(
            [{"role": "system", "content": ANSWERER_SYSTEM_PROMPT}, {"role": "user", "content": query}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "AnswererResponse", "schema": AnswererResponse.model_json_schema()},
            },
        )
        content = str(message.get("content") or "")
        try:
            return AnswererResponse.model_validate_json(content)
        except ValidationError:
            # Some backends wrap JSON in prose or code fences; take the outermost object.
            start, end = content.find("{"), content.rfind("}")
            if start == -1 or end <= start:
                raise
            return AnswererResponse.model_validate(json.loads(content[start : end + 1]))

    async def answer_question(self) -> None:
        """Process the player's question and generate an answer."""
        query = ANSWERER_QUERY_TEMPLATE.format(
            answer=self.state.answer, next_question=self.state.next_question, category=self.state.category
        )
        # NOTE: We can also ground the answerer with a search tool.
        # But it would make the example too complicated for now.
        try:
            answerer_response = await self._call_answerer(query)
            console.print(f"[bold red]Answerer (Turn {self.state.turn_index}):[/bold red] {answerer_response}")
            turn = Turn(question=self.state.next_question, response=answerer_response.yes_or_no)
            correct = answerer_response.correct
        except (ValidationError, ValueError) as e:
            console.print(f"[bold red]Answerer Response Format Error: {e}[/bold red]")
            # Assuming n/a
            turn = Turn(question=self.state.next_question, response="n/a")
            correct = False
        self.state.interactions.append(turn)
        self.state.next_question = ""  # Reset the next question
        self.state.correct = correct

    def game_should_continue(self) -> Literal["game_over", "next_turn"]:
        """Determine if the game should continue or end.

        Returns:
            "game_over" if the game is finished, "next_turn" otherwise.
        """
        if self.state.correct:
            console.print(f"[bold red]Correct! You win![/bold red]")
            return "game_over"
        elif self.state.turn_index >= 20:
            console.print(
                f"[bold red]You've asked 20 questions and still haven't guessed the entity. You lose![/bold red]"
            )
            return "game_over"
        else:
            self.state.turn_index += 1
            console.print(f"[bold purple]Continue with turn #{self.state.turn_index}...[/bold purple]")
            return "next_turn"

    def finish(self) -> None:
        """Handle game completion."""
        console.print("The flow has reached the finished state.")
