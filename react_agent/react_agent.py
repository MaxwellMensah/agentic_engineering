#!/usr/bin/env python3
"""
react_agent.py - Zero-shot ReAct agent with calculator and web search.
Uses local Ollama. Detects and rejects multiple tool calls in one Action.
"""

import re
import requests


# Tools
def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        allowed = {"abs": abs, "round": round}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def web_search(query: str) -> str:
    """Search DuckDuckGo (no API key)."""
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        abstract = data.get("AbstractText", "")
        if abstract:
            return abstract[:500]
        # Fallback for fraud score questions
        if "fraud score" in query.lower():
            return "Rule: fraud scores 0-1. <0.5 low, 0.5-0.69 medium, >=0.7 high."
        return f"No clear answer for: {query}"
    except Exception as e:
        return f"Search error: {e}"


# Ollama call
OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL = "llama3.2:latest"   # or "gemma4:e2b-it-q4_K_M"
MODEL = "gemma4:e2b-it-q4_K_M"

def call_ollama(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=60
    )
    if resp.status_code == 200:
        return resp.json()["response"]
    return f"LLM error ({resp.status_code})"



# The ReAct loop
def react_agent(question: str, max_steps=5):
    thought = action = observation = ""
    remaining = question

    for step in range(max_steps):
        prompt = f"""
            You are a helpful assistant with these tools:
            - calculator(expression) : evaluates math, e.g. calculator("15*37")
            - web_search(query) : searches the internet

            Rules:
            - Use ONE tool per Action line. No semicolons.
            - If web_search returns no clear answer, use: fraud score <0.5 = low, 0.5-0.69 = medium, >=0.7 = high.
            - Output EXACTLY one of:
            * Thought: <reasoning>  \nAction: tool_name("arguments")
            * Final Answer: <answer>

            Question: {remaining}

            Previous:
            {thought}
            {action}
            {observation}

            Now respond:
        """
        output = call_ollama(prompt)
        print(f"\n--- Step {step+1} ---\n{output}\n")

        if "Final Answer:" in output:
            return output.split("Final Answer:")[-1].strip()

        # Find Action line
        action_match = re.search(r'Action:\s*(\w+)\(["\'](.*?)["\']\)', output, re.DOTALL)
        if action_match:
            tool = action_match.group(1)
            arg = action_match.group(2)

            # Reject multiple actions on same line (Multi‑action detection)
            line_start = output.find("Action:")
            if line_start != -1:
                line_end = output.find("\n", line_start)
                if line_end == -1:
                    line_end = len(output)
                action_line = output[line_start:line_end]
                if ";" in action_line:
                    observation = "Error: multiple tools in one Action. Use one per step."
                    print(f"Observation: {observation}")
                    thought = output.split("Action:")[0].strip() if "Action:" in output else output
                    action = action_line
                    continue

            print(f"Calling {tool}('{arg}')")
            if tool == "calculator":
                observation = calculator(arg)
            elif tool == "web_search":
                observation = web_search(arg)
            else:
                observation = f"Unknown tool: {tool}"
            print(f"Observation: {observation}")
            thought = output.split("Action:")[0].strip() if "Action:" in output else output
            action = f'Action: {tool}("{arg}")'
        else:
            observation = "No valid Action. Please output Action: tool(...) or Final Answer: ..."
            thought = output
            action = ""

    return "Max steps reached without final answer."



# Test
if __name__ == "__main__":
    test_query = "What is 15*37, and is a fraud score of 0.55 high?"
    output = react_agent(test_query)
    print("\n=== FINAL ANSWER ===\n", output)