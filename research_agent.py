"""
Web Research Agent (ReAct pattern: Plan -> Act -> Report)

Flow:
    start -> questions generation (Claude) -> web search (Tavily) -> final report (Claude)

Setup:
    pip install anthropic tavily-python
    Put your keys in a .env file next to this script (see .env.example):
        ANTHROPIC_API_KEY=your-anthropic-key
        TAVILY_API_KEY=your-tavily-key
    or export them as environment variables.

Usage:
    python research_agent.py "Climate Change"
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field

import anthropic
from tavily import TavilyClient


def load_env_file(path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")) -> None:
    """Load KEY=VALUE lines from a .env file without overriding existing variables."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
NUM_QUESTIONS = 6
RESULTS_PER_QUESTION = 5


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    """A single web search result."""
    title: str
    url: str
    content: str


@dataclass
class ResearchItem:
    """Everything the agent learns about one research question."""
    question: str
    rationale: str = ""
    sources: list[Source] = field(default_factory=list)
    key_points: str = ""


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class ResearchAgent:
    """Plans research questions, searches the web, and compiles a report."""

    def __init__(self, topic: str, model: str = MODEL) -> None:
        self.topic = topic
        self.model = model
        self.llm = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.search_tool = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        self.items: list[ResearchItem] = []

    # ----- LLM helper ------------------------------------------------------- #
    def _ask(self, prompt: str, system: str = "", max_tokens: int = 1500) -> str:
        """Send one prompt to Claude and return the text of the reply."""
        response = self.llm.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()

    # ----- Phase 1: Planning (LLM reasoning) -------------------------------- #
    def plan(self) -> None:
        """Ask the LLM for research questions, each with a short rationale."""
        system = (
            "You are a research planner. You break a topic into focused, "
            "non-overlapping research questions. Reply with JSON only."
        )
        prompt = (
            f"Topic: {self.topic}\n\n"
            f"Generate {NUM_QUESTIONS} well-structured research questions that cover "
            "different aspects of this topic (background, causes, current state, "
            "impact, solutions, future outlook, and so on).\n\n"
            'Return a JSON array of objects with the keys "question" and "rationale", '
            "where the rationale is one sentence explaining why the question matters. "
            "Do not add any text outside the JSON."
        )
        self.items = self._parse_plan(self._ask(prompt, system))
        if not self.items:
            raise RuntimeError("The LLM did not return any research questions.")

    @staticmethod
    def _parse_plan(raw: str) -> list[ResearchItem]:
        """Parse the planner output, tolerating code fences and plain lists."""
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            data = json.loads(cleaned)
            return [ResearchItem(d["question"], d.get("rationale", "")) for d in data]
        except (json.JSONDecodeError, KeyError, TypeError):
            lines = [re.sub(r"^\s*[\d\-\*\.\)]+\s*", "", ln) for ln in cleaned.splitlines()]
            return [ResearchItem(ln.strip()) for ln in lines if ln.strip().endswith("?")]

    # ----- Phase 2: Acting (web search tool) -------------------------------- #
    def act(self, item: ResearchItem) -> None:
        """Search the web for one question and store title, url, and content."""
        response = self.search_tool.search(
            query=item.question,
            max_results=RESULTS_PER_QUESTION,
            search_depth="advanced",
        )
        item.sources = [
            Source(r.get("title", "Untitled"), r.get("url", ""), r.get("content", ""))
            for r in response.get("results", [])
        ]

    def observe(self, item: ResearchItem) -> None:
        """Condense the raw search results into key points using the LLM."""
        if not item.sources:
            item.key_points = "- No relevant information was found for this question."
            return

        evidence = "\n\n".join(
            f"[{i}] {s.title}\n{s.content}" for i, s in enumerate(item.sources, start=1)
        )
        prompt = (
            f"Research question: {item.question}\n\n"
            f"Search results:\n{evidence}\n\n"
            "Extract 3 to 5 key points that answer the question. Use only the "
            "information in the search results. Cite the result numbers in square "
            "brackets, for example [1][3]. Return a markdown bullet list only."
        )
        item.key_points = self._ask(
            prompt,
            system="You are a careful research assistant who never invents facts.",
        )

    # ----- Phase 3: Report generation --------------------------------------- #
    def compile_report(self) -> str:
        """Assemble the final markdown report."""
        system = (
            "You are a professional report writer. Write clear, plain prose "
            "without em-dashes."
        )
        questions = "\n".join(f"- {it.question}" for it in self.items)
        findings = "\n\n".join(f"{it.question}\n{it.key_points}" for it in self.items)

        introduction = self._ask(
            f"Write a short introduction (one paragraph) for a research report on "
            f"'{self.topic}'. It should preview these questions:\n{questions}",
            system,
        )
        conclusion = self._ask(
            f"Write a short conclusion (one paragraph) for a research report on "
            f"'{self.topic}', synthesizing these findings:\n\n{findings}",
            system,
        )

        parts = [f"# Research Report: {self.topic}", "## Introduction", introduction]
        for n, item in enumerate(self.items, start=1):
            sources = "\n".join(
                f"{i}. [{s.title}]({s.url})" for i, s in enumerate(item.sources, start=1)
            )
            parts += [
                f"## {n}. {item.question}",
                item.key_points,
                f"**Sources**\n\n{sources}" if sources else "",
            ]
        parts += ["## Conclusion", conclusion]
        return "\n\n".join(p for p in parts if p)

    # ----- Orchestration ---------------------------------------------------- #
    def run(self) -> str:
        """Run the full Plan -> Act -> Observe -> Report loop."""
        print(f"\n[Plan] Generating research questions for: {self.topic}")
        self.plan()

        for n, item in enumerate(self.items, start=1):
            print(f"\n--- Question {n}/{len(self.items)} ---")
            print(f"Thought: {item.rationale or 'Gather evidence for this question.'}")
            print(f"Action: web_search({item.question!r})")
            try:
                self.act(item)
            except Exception as exc:  # keep going if one search fails
                print(f"Observation: search failed ({exc})")
                continue
            print(f"Observation: {len(item.sources)} results retrieved")
            self.observe(item)

        print("\n[Report] Compiling final report")
        return self.compile_report()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="ReAct web research agent")
    parser.add_argument("topic", help="Topic to research, e.g. 'Climate Change'")
    parser.add_argument("-o", "--output", help="Output markdown file path")
    args = parser.parse_args()

    for key in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
        if not os.getenv(key):
            raise SystemExit(f"Missing {key}: add it to .env or export it as an environment variable")

    report = ResearchAgent(args.topic).run()

    slug = re.sub(r"[^a-z0-9]+", "_", args.topic.lower()).strip("_")
    path = args.output or f"report_{slug}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nDone. Report saved to {path}\n")
    print(report)


if __name__ == "__main__":
    main()
