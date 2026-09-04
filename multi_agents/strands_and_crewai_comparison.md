## Architectural Comparative Breakdown

<p align="center">
  <img src="https://img.shields.io/badge/Strands-Agent Framework-darkgreen?logo=strands" alt="Strands" />
  <img src="https://img.shields.io/badge/CrewAI-Orchestration-orange?logo=crewai" alt="CrewAI" />
</p>

| Dimension | Strands Agents | CrewAI |
| --- | --- | --- |
| **Task Delegation Style** | **Model-Driven & Dynamic.** Multi-agent orchestration is handled via sub-agents as tools, explicit Graphs, or Swarm handoffs. Agents decide dynamically when to invoke sub-agents at runtime. | **Role & Task-Driven.** Declarative pipeline where agents are assigned specific `Task` objects. Execution follows a rigid sequential or hierarchical process flow. |
| **Tool-Use API** | **Native Pythonic & MCP Support.** Uses standard Python function decorators (`@tool`) and has native Model Context Protocol (MCP) integration. Whole `Agent` instances automatically convert into callable tools. | **Decorator & Class-Based.** Tools use the `@tool` decorator or extend `BaseTool`. Highly integrated with the `crewai_tools` package and LangChain tool ecosystem. |
| **Memory Handling** | **Process & Provider Level.** Managed via session context, Bedrock Knowledge Bases, Valkey/Redis session stores, or shared context objects across Swarm/Graph nodes. | **Built-in Multi-Layer Memory.** Out-of-the-box support for Short-Term Memory, Long-Term Memory (vector databases like Chroma), and Entity Memory. |