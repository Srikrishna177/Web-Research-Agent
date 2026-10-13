# Web Research Agent using the ReAct Pattern

## Overview

This project implements a research agent that takes a topic, plans a set of research questions with an LLM (Claude), gathers evidence from the web with Tavily, and compiles a structured markdown report.

Program flow:

```
start -> questions generation -> web search -> final report
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then put your real keys in .env
python research_agent.py "Climate Change"
```

## How the LLM is used for reasoning

Claude is used at four points, and each call has a narrow job:

1. **Planning.** The agent asks Claude for 6 research questions on the topic. Each question comes with a one-sentence rationale explaining why it matters. The reply is requested as JSON so it can be parsed reliably.
2. **Observation.** After the web search returns raw results, Claude extracts 3 to 5 key points per question. It is told to use only the supplied search results and to cite result numbers, which keeps the answers grounded and limits hallucination.
3. **Introduction.** Claude writes an introduction that previews the questions.
4. **Conclusion.** Claude synthesizes the key points from all questions into a conclusion.

### The reasoning step

The reasoning (the "Thought" in ReAct) happens in the planning phase. Instead of searching for the topic directly, the LLM first decomposes it into focused sub-questions covering different aspects. Each rationale is logged as the agent's thought before it acts. The loop for every question is:

```
Thought      -> why this question matters (from the LLM plan)
Action       -> web_search(question) using Tavily
Observation  -> results retrieved, condensed into key points by the LLM
```

Sample console trace:

```
Thought: Understanding the causes is the foundation for any policy discussion.
Action: web_search('What are the main causes of climate change?')
Observation: 5 results retrieved
```

## Patterns used

- **Planning pattern:** questions are generated first, then searched.
- **Tool-use pattern:** the agent uses two tools, the LLM for reasoning and writing, and Tavily for web search.

## Code walkthrough

| Component | Purpose |
|---|---|
| `Source` | Dataclass holding the title, url, and content of one search result. |
| `ResearchItem` | Dataclass holding a question, its rationale, its sources, and its key points. |
| `ResearchAgent.__init__` | Creates the Claude and Tavily clients and stores the topic. |
| `_ask` | Small helper that sends one prompt to Claude and returns the text reply. |
| `plan` and `_parse_plan` | Generate the questions and parse the JSON, with a fallback for plain numbered lists. |
| `act` | Runs a Tavily search for one question and stores title, url, and content of every result. |
| `observe` | Uses Claude to turn raw results into cited key points. |
| `compile_report` | Builds the final markdown: title, introduction, one section per question with sources, and a conclusion. |
| `run` | Orchestrates the full loop and prints the Thought, Action, Observation trace. |
| `load_env_file` | Loads API keys from a `.env` file next to the script (never committed). |
| `main` | Command line entry point that validates API keys and saves the report to a file. |

## Design choices

- **Separation of concerns:** planning, searching, summarizing, and reporting are separate methods, so each can be tested or swapped independently.
- **Grounded summaries:** the observation prompt restricts Claude to the retrieved content.
- **Robustness:** a failed search for one question does not stop the whole run, and the JSON parser handles code fences and plain-text lists.
- **Configurable model:** set `CLAUDE_MODEL` to switch models without touching the code.
