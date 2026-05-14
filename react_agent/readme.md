# ReAct Agent with Local Ollama

A lightweight, zero‑shot ReAct (Reasoning + Acting) agent that uses local LLMs (Ollama) to answer questions by calling tools: a calculator and a web search. The agent runs entirely on your machine – no API keys, no cloud dependencies.

## Features

- **Tool use**: calculator (safe `eval`) and web search (DuckDuckGo, no API key).
- **ReAct loop**: Thought → Action → Observation → (repeat) → Final Answer.
- **Multi‑action detection**: rejects attempts to call two tools on one line (e.g., `Action: calculator(...); web_search(...)`).
- **Fallback rules**: when web search fails for fraud‑score questions, a built‑in rule provides the answer.
- **Model agnostic**: works with any Ollama model (tested with `gemma4:e2b` and `llama3.2:latest`).

## How It Works

The agent alternates between reasoning (“Thought”) and executing tools (“Action”).  
It stops when it outputs “Final Answer”.  

Example flow for the question:  
*“What is 15*37, and is a fraud score of 0.55 high?”*

1. **Thought**: I need to calculate 15×37.  
   **Action**: `calculator("15*37")`  
   **Observation**: `555`

2. **Thought**: Now I need to know if 0.55 is high.  
   **Action**: `web_search("fraud score 0.55 high")`  
   **Observation**: (if no web result) *Rule: fraud scores 0-1. <0.5 low, 0.5-0.69 medium, >=0.7 high.*

3. **Thought**: 0.55 is in the medium range.  
   **Final Answer**: `15*37 = 555. A fraud score of 0.55 is medium (not high).`

## Fallback Mechanisms

### 1. Multi‑action detection (code level)

If the LLM outputs an `Action:` line containing a semicolon (e.g., `Action: calculator("..."); web_search("...")`), the agent rejects it, injects an error observation, and retries – forcing one tool per step.

### 2. Fraud score fallback (tool level)

Inside the `web_search()` function, if DuckDuckGo returns no abstract and the query contains `"fraud score"`, the function returns a hardcoded rule instead of “No clear answer”. This guarantees the agent has the information to classify scores correctly, even without internet access.

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com) running locally
- `requests` library (`pip install requests`)

## Installation & Usage

1. **Install Ollama** and pull a model (e.g., `ollama pull gemma4:e2b-it-q4_K_M`).
2. **Start Ollama** in a terminal: `ollama serve`
3. **Clone this repository** (or save `react_agent.py`).
4. **Install dependencies**: `pip install requests`
5. **Run the agent**: `python react_agent.py`

You can change the model by editing the `MODEL` variable at the top of the script.

## Example Output

```bash
--- Step 1 ---
Action: calculator("15*37")

Calling calculator('15*37')
Observation: 555

--- Step 2 ---
Action: web_search("fraud score 0.55 high")

Calling web_search('fraud score 0.55 high')
Observation: Rule: fraud scores 0-1. <0.5 low, 0.5-0.69 medium, >=0.7 high.

--- Step 3 ---
Final Answer: 15*37 = 555. A fraud score of 0.55 is medium.

=== FINAL ANSWER ===
15*37 = 555. A fraud score of 0.55 is medium.