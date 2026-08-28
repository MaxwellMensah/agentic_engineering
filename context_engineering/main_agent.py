from __future__ import annotations

import ast
import math
import operator as op
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from suppress_warnings import silence_warnings

silence_warnings()

from context_builder import ContextBuilder
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    SummarizationMiddleware,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain.tools.tool_node import ToolCallRequest
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from tavily import TavilyClient

# Environment Configuration
load_dotenv()

# PRODUCTION CONSTANTS
MAX_STEPS = 6
CONTEXT_TOKEN_BUDGET = 1200
CONTEXT_QUALITY_THRESHOLD = 0.60
SUMMARY_TRIGGER_TOKENS = 4000
SUMMARY_KEEP_MESSAGES = 10
MAX_TOOL_OUTPUT_CHARS = 5000

# Gemini Model Orchestration
agent_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=6,
)
heavy_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    max_retries=6,
)
compression_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

context_builder = ContextBuilder(
    token_budget=CONTEXT_TOKEN_BUDGET,
    quality_threshold=CONTEXT_QUALITY_THRESHOLD,
    compression_llm=compression_llm,
    verbose=True,
)

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))


@dataclass
class Context:
    """Production runtime context passed through the middleware pipeline."""

    user_role: str = "user"
    environment: str = "production"
    expertise_level: str = "intermediate"


# TOOLS
@tool
def search(query: str) -> str:
    """Use for current events, news, people, political figures, dates, and general web searches."""
    try:
        response = tavily_client.search(
            query=query, search_depth="basic", max_results=5, include_answer=False
        )
    except Exception as exc:  # noqa
        return f"Search failed: {exc}"

    results = response.get("results", [])
    if not results:
        return "No relevant search results found."

    chunks = []
    for result in results:
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")
        if content:
            chunks.append(f"TITLE: {title}\nSOURCE: {url}\nCONTENT:\n{content}")

    if not chunks:
        return "Search returned no usable content."

    optimized_context = context_builder.build(chunks=chunks, query=query)
    return (
        optimized_context
        if optimized_context
        else "Search results found, but none fit the token budget."
    )


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely using AST parsing."""
    allowed_ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }
    allowed_funcs = {
        name: getattr(math, name) for name in dir(math) if not name.startswith("_")
    }
    allowed_funcs.update({"abs": abs, "round": round, "pow": pow})

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return allowed_ops[type(node.op)](_eval(node.operand))
        elif isinstance(node, ast.Call):
            func = allowed_funcs[node.func.id]
            args = [_eval(arg) for arg in node.args]
            return func(*args)
        elif isinstance(node, ast.Name):
            return allowed_funcs[node.id]
        raise ValueError("Unsupported math structure.")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree))
    except Exception as exc:  # noqa
        return f"Calculator error: {exc}"


@tool
def weather_checker(location: str) -> str:
    """Get current weather for a city or location."""
    # this is for demonstration purposes; in a real implementation, you would call a weather API for accurate forecasts
    return (
        f"Weather for {location}: 28°C, partly cloudy, humidity 65%, wind 12 km/h NE."
    )


TOOLS = [search, calculator, weather_checker]


# PRODUCTION MIDDLEWARE PIPELINE
@dynamic_prompt
def adaptive_system_prompt(request: ModelRequest) -> str:
    """Step 1: Construct system persona from request runtime context."""
    context = request.runtime.context
    message_count = len(request.messages)

    prompt = f"""You are an accurate agentic assistant.

                User Context:
                - Role: {context.user_role}
                - Expertise: {context.expertise_level}
                - Environment: {context.environment}

                Rules:
                1. Always invoke tools directly when real-world facts or math calculations are required.
                2. Provide a clear, direct answer once tool results are returned.

                Current conversation contains approximately {message_count} messages."""

    if context.expertise_level == "beginner":
        prompt += "\n- Explain technical terms in accessible words."
    elif context.expertise_level == "advanced":
        prompt += "\n- Use precise terminology and emphasize trade-offs."

    if context.environment == "production":
        prompt += "\n- Production policy: Be conservative, precise, and never invent missing facts."

    return prompt


@wrap_model_call
def input_guardrail(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Inspect request messages for injection vectors or restricted actions."""
    forbidden_patterns = [
        "ignore previous instructions",
        "system prompt override",
        "drop database",
    ]
    for msg in request.messages:
        content_str = str(getattr(msg, "content", "")).lower()
        if any(pattern in content_str for pattern in forbidden_patterns):
            print("[Guardrail] Prompt injection pattern detected.")
            raise ValueError(
                "Security Violation: Request contains prohibited prompt pattern."
            )
    return handler(request)


@wrap_model_call
def dynamic_model_selection(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Step 3: Route context dynamically based on conversation depth."""
    if len(request.messages) > 14:
        print("[ModelPolicy] High message volume → Switching to Gemini Pro.")
        request = request.override(model=heavy_llm)
    return handler(request)


@wrap_model_call
def role_based_tools(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Enforce role-based tool restrictions (RBAC)."""
    role = request.runtime.context.user_role
    available_tools = list(request.tools)

    if role == "viewer":
        available_tools = [
            t for t in available_tools if t.name in {"search", "weather_checker"}
        ]
        print("[ToolPolicy] viewer role → calculator tool removed")
    elif role == "calculator_only":
        available_tools = [t for t in available_tools if t.name == "calculator"]
        print("[ToolPolicy] calculator_only role → search and weather removed")

    return handler(request.override(tools=available_tools))


@wrap_model_call
def step_limiter(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Enforce upper bound on execution turns (Runs AFTER RBAC)."""

    def is_ai_message(msg) -> bool:
        if isinstance(msg, dict):
            return msg.get("type") == "ai" or msg.get("role") == "assistant"
        return getattr(msg, "type", "") == "ai"

    steps_taken = len([m for m in request.messages if is_ai_message(m)])
    print(f"[StepLimiter] Turn {steps_taken + 1}/{MAX_STEPS}")

    if steps_taken >= MAX_STEPS:
        print("[StepLimiter] Step limit reached. Disabling tool access.")
        request = request.override(tools=[])

    return handler(request)


@wrap_model_call
def production_telemetry(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Log latency and token usage per LLM invocation."""
    start_time = time.perf_counter()
    response = handler(request)
    elapsed = time.perf_counter() - start_time

    usage_metadata = getattr(response, "usage_metadata", {})
    prompt_tokens = (
        usage_metadata.get("input_tokens", 0)
        if isinstance(usage_metadata, dict)
        else getattr(usage_metadata, "input_tokens", 0)
    )
    completion_tokens = (
        usage_metadata.get("output_tokens", 0)
        if isinstance(usage_metadata, dict)
        else getattr(usage_metadata, "output_tokens", 0)
    )

    print(
        f"[Telemetry] Duration: {elapsed:.3f}s | Prompt Tokens: {prompt_tokens} | Completion Tokens: {completion_tokens}"
    )
    return response


class ToolOutputCompressionMiddleware(AgentMiddleware):
    def __init__(self, *, threshold: int = 5000) -> None:
        super().__init__()
        self.threshold = threshold

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        response = handler(request)

        if not isinstance(response, ToolMessage) or not isinstance(
            response.content, str
        ):
            return response

        if (
            response.content.startswith("[compressed tool output]")
            or len(response.content) <= self.threshold
        ):
            return response

        print(
            f"[ToolCompression] {response.name}: {len(response.content)} chars → compressing"
        )
        prompt = f"Summarize facts, numbers, dates, and claims concisely:\n\n{response.content}"
        compressed = compression_llm.invoke(prompt).content

        return ToolMessage(
            content=f"[compressed tool output]\n{compressed}",
            tool_call_id=response.tool_call_id,
            name=response.name,
        )


class ToolErrorMiddleware(AgentMiddleware):
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        try:
            return handler(request)
        except Exception as exc:  # noqa
            tool_name = request.tool_call["name"]
            print(f"[ToolError] {tool_name}: {exc}")
            return ToolMessage(
                content=f"The {tool_name} tool failed with error: {exc}. Adjust inputs if possible.",
                tool_call_id=request.tool_call["id"],
            )


# AGENT INITIALIZATION
checkpointer = MemorySaver()

agent = create_agent(
    model=agent_llm,
    tools=TOOLS,
    checkpointer=checkpointer,
    middleware=[
        adaptive_system_prompt,         # System Prompt Construction
        input_guardrail,                # Input Security & Guardrails
        dynamic_model_selection,        # Model Routing
        role_based_tools,               # RBAC Tool Filtering
        step_limiter,                   # Step Limiter (Strips tools if step limit hit)
        production_telemetry,           # Observability & Token Metrics
        ToolOutputCompressionMiddleware(threshold=MAX_TOOL_OUTPUT_CHARS),
        ToolErrorMiddleware(),
        SummarizationMiddleware(
            model=compression_llm,
            trigger=("tokens", SUMMARY_TRIGGER_TOKENS),
            keep=("messages", SUMMARY_KEEP_MESSAGES),
        ),
    ],
    context_schema=Context,
)


def run(
    query: str,
    *,
    thread_id: str = "default_session",
    user_role: str = "user",
    environment: str = "production",
    expertise_level: str = "intermediate",
):
    print(f"\n{'=' * 70}\nUSER: {query}\n{'=' * 70}")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        context=Context(
            user_role=user_role,
            environment=environment,
            expertise_level=expertise_level,
        ),
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 15,
        },
    )

    final_msg = result["messages"][-1]
    if isinstance(final_msg.content, list):
        clean_text = "".join(
            block["text"]
            for block in final_msg.content
            if isinstance(block, dict) and "text" in block
        )
    else:
        clean_text = str(final_msg.content)

    print(f"\n{'=' * 70}\nFINAL ANSWER\n{clean_text}\n{'=' * 70}")
    return result


if __name__ == "__main__":
    run(
        "Who is the current president of Ghana in 2026?",
        thread_id="sess_search",
        expertise_level="advanced",
    )

    run(
        "What is 125 * 48 and then calculate its square root?",
        thread_id="sess_math",
        expertise_level="advanced",
    )

    run(
        "What is the weather in Accra?",
        thread_id="sess_weather",
        expertise_level="beginner",
    )

    print(f"\n{'=' * 70}\nCONTEXT ENGINEERING AUDIT LOG\n{'=' * 70}")
    print(context_builder.audit_json())
