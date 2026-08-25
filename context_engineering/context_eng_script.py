import math
import os
from collections.abc import Callable
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    SummarizationMiddleware,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from tavily import TavilyClient

# TOOLS
# Initialize the search client
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))


@tool(parse_docstring=True)
def search(query: str) -> str:
    """Search the web for real-time information, news, and factual verification.

    Args:
        query: The topic or query to look up.
    """
    try:
        # Tavily handles query optimization, semantic reranking, and context extraction
        context = tavily_client.search(
            query=query, search_depth="basic", max_tokens=1500
        )
        return context if context else "No relevant context found."
    except Exception as e:  # noqa
        return f"Tavily search execution failed: {e}"


@tool(parse_docstring=True)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports standard math functions: sqrt, sin, cos, log, pow, etc.

    Args:
        expression: A valid Python math expression, e.g. 'sqrt(144) + 56'
    """
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:  # noqa
        return f"Error evaluating '{expression}': {e}"


@tool(parse_docstring=True)
def weather_checker(location: str) -> str:
    """Get the current weather for a given city or location.

    Args:
        location: City name or location string, e.g. 'Accra, Ghana'
    """
    return (
        f"[Weather — {location}]: 28°C, partly cloudy, humidity 65%, wind 12 km/h NE."
    )


TOOLS = [search, calculator, weather_checker]


# RUNTIME CONTEXT
@dataclass
class Context:
    user_role: str = "user"
    environment: str = "production"


# MODELS
agent_llm = ChatOllama(
    model="gemma4:e2b-it-q4_K_M",
    temperature=0,
    streaming=True,
)

compressor_llm = ChatOllama(
    model="qwen2.5-coder:1.5b",
    temperature=0,
)


COMPRESS_THRESHOLD = 300
MAX_STEPS = 5


# MIDDLEWARE
@dynamic_prompt
def adaptive_system_prompt(request: ModelRequest) -> str:
    message_count = len(request.messages)

    base = (
        "You are a helpful ReAct agent with access to:\n"
        "- search: for current web information\n"
        "- calculator: for math expressions\n"
        "- weather_checker: for live weather data\n\n"
        "Rules:\n"
        "1. Call each tool ONCE per question. Never call the same tool twice.\n"
        "2. After collecting tool results, give a final answer immediately.\n"
        "3. Do NOT loop or re-verify. Trust the tool output.\n"
        "Think step-by-step but be decisive."
    )

    if message_count > 10:
        base += "\n\nConversation is long — give your final answer NOW."

    return base


@wrap_model_call
def step_limiter(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:

    # Helper to safely check if a message is from the AI
    def is_ai_msg(m):
        if isinstance(m, dict):
            return m.get("type") == "ai" or m.get("role") == "ai"
        return getattr(m, "type", "") == "ai"

    # Count steps using the safe helper
    steps_taken = sum(1 for m in request.messages if is_ai_msg(m))
    print(f"[step {steps_taken + 1}/{MAX_STEPS}]")

    if steps_taken >= MAX_STEPS:
        print("[max steps reached — stripping tools to force final answer]")
        request = request.override(tools=[])

    return handler(request)


@wrap_model_call
def role_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    if request.runtime.context.user_role == "viewer":
        tools = [t for t in request.tools if t.name != "calculator"]
        request = request.override(tools=tools)
    return handler(request)


@wrap_model_call
def production_guardrail(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    if request.runtime.context.environment == "production":
        messages = [
            SystemMessage(content="Note: production env — be precise and concise."),
            *request.messages,
        ]
        request = request.override(messages=messages)
    return handler(request)


class ToolOutputCompressionMiddleware(AgentMiddleware):
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        response = handler(request)

        if not isinstance(response, ToolMessage):
            return response

        content = response.content

        if not isinstance(content, str) or len(content) <= COMPRESS_THRESHOLD:
            return response

        print(f"[compressing '{response.name}' output: {len(content)} chars → ...]")
        compressed = compressor_llm.invoke(
            f"Compress the following {response.name or 'tool'} output to its essential "
            f"facts in 2-3 sentences. Preserve all numbers, names, dates, and critical "
            f"values exactly:\n\n{content}"
        )

        return ToolMessage(
            content=f"[compressed] {compressed.content}",
            tool_call_id=response.tool_call_id,
            name=response.name,
        )


# CREATE AGENT
agent = create_agent(
    model=agent_llm,
    tools=TOOLS,
    middleware=[
        adaptive_system_prompt,
        step_limiter,
        role_based_tools,
        production_guardrail,
        ToolOutputCompressionMiddleware(),
        SummarizationMiddleware(
            model=compressor_llm,
            trigger={"tokens": 4_000},
            keep=("messages", 10),
        ),
    ],
    context_schema=Context,
)


# RUN
if __name__ == "__main__":
    print("Running agent...\n")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Who is the current president of Ghana in 2026?",
                }
            ]
        },
        context=Context(user_role="user", environment="production"),
        config={"recursion_limit": 15},
    )

    print("\n=== Final conversation ===\n")
    for msg in result["messages"]:
        if isinstance(msg, dict):
            role = msg.get("type", msg.get("role", "unknown")).upper()
            content = msg.get("content", "")
        else:
            role = type(msg).__name__.replace("Message", "").upper()
            content = msg.content

        content_str = content if isinstance(content, str) else str(content)
        if len(content_str) > 400:
            content_str = content_str[:400] + "..."
        print(f"[{role}]: {content_str}\n")
