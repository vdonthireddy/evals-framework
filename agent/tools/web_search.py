"""Web search tool with simulated search results for deterministic testing."""

from __future__ import annotations

from typing import Any

from agent.tools.base import BaseTool, ToolResult

# Hardcoded search results covering diverse topics
_SEARCH_DATABASE: dict[str, list[dict[str, str]]] = {
    "ai": [
        {"title": "The State of AI in 2025", "snippet": "Agentic AI systems are transforming software development with autonomous coding assistants and multi-agent workflows.", "url": "https://example.com/state-of-ai-2025"},
        {"title": "GPT-5 and the Next Generation of Language Models", "snippet": "OpenAI's latest model demonstrates unprecedented reasoning capabilities across complex multi-step tasks.", "url": "https://example.com/gpt5-next-gen"},
        {"title": "Google DeepMind's Gemini 2.5 Pro", "snippet": "Gemini 2.5 Pro achieves state-of-the-art performance on agentic coding benchmarks.", "url": "https://example.com/gemini-25-pro"},
    ],
    "machine learning": [
        {"title": "Machine Learning Fundamentals", "snippet": "A comprehensive guide to supervised, unsupervised, and reinforcement learning paradigms.", "url": "https://example.com/ml-fundamentals"},
        {"title": "Transformers Explained", "snippet": "The transformer architecture has become the backbone of modern NLP and computer vision.", "url": "https://example.com/transformers-explained"},
    ],
    "climate change": [
        {"title": "Global Temperature Rise Accelerates", "snippet": "2025 marks the hottest year on record, with global temperatures 1.5°C above pre-industrial levels.", "url": "https://example.com/climate-2025"},
        {"title": "Renewable Energy Surpasses Fossil Fuels", "snippet": "Solar and wind energy now account for 45% of global electricity generation.", "url": "https://example.com/renewable-milestone"},
    ],
    "python programming": [
        {"title": "Python 3.13 New Features", "snippet": "Python 3.13 introduces a JIT compiler, improved error messages, and new typing features.", "url": "https://example.com/python-313"},
        {"title": "AsyncIO Best Practices", "snippet": "Learn how to write efficient asynchronous Python code with asyncio and modern patterns.", "url": "https://example.com/asyncio-best-practices"},
        {"title": "Pydantic V2 Migration Guide", "snippet": "Pydantic V2 is 5-50x faster with a new Rust-based core validator.", "url": "https://example.com/pydantic-v2"},
    ],
    "space exploration": [
        {"title": "Artemis III Moon Landing", "snippet": "NASA's Artemis III mission successfully lands astronauts on the lunar south pole.", "url": "https://example.com/artemis-iii"},
        {"title": "SpaceX Starship Orbital Flight", "snippet": "Starship completes its first full orbital flight and booster catch.", "url": "https://example.com/starship-orbital"},
    ],
    "cooking": [
        {"title": "Best Pasta Recipes for Beginners", "snippet": "Simple yet delicious pasta recipes including aglio e olio, carbonara, and cacio e pepe.", "url": "https://example.com/pasta-recipes"},
        {"title": "Sourdough Bread Masterclass", "snippet": "A step-by-step guide to making artisanal sourdough bread at home.", "url": "https://example.com/sourdough-guide"},
    ],
    "history": [
        {"title": "The Fall of the Roman Empire", "snippet": "Exploring the political, economic, and military factors that led to Rome's decline.", "url": "https://example.com/roman-empire-fall"},
        {"title": "The Renaissance: A Cultural Revolution", "snippet": "How art, science, and philosophy transformed Europe in the 14th-17th centuries.", "url": "https://example.com/renaissance"},
    ],
    "quantum computing": [
        {"title": "Quantum Supremacy Achieved", "snippet": "Google's Willow chip demonstrates quantum error correction below threshold.", "url": "https://example.com/quantum-supremacy"},
        {"title": "Quantum Computing for Beginners", "snippet": "Understanding qubits, superposition, and entanglement in simple terms.", "url": "https://example.com/quantum-beginners"},
    ],
    "health": [
        {"title": "Benefits of Regular Exercise", "snippet": "30 minutes of daily exercise reduces the risk of heart disease by 40%.", "url": "https://example.com/exercise-benefits"},
        {"title": "Mediterranean Diet and Longevity", "snippet": "Studies show the Mediterranean diet extends life expectancy by up to 7 years.", "url": "https://example.com/mediterranean-diet"},
    ],
    "cryptocurrency": [
        {"title": "Bitcoin Reaches New All-Time High", "snippet": "Bitcoin surpasses $150,000 as institutional adoption accelerates.", "url": "https://example.com/bitcoin-ath"},
        {"title": "Ethereum's Layer 2 Ecosystem", "snippet": "Layer 2 solutions have reduced Ethereum gas fees by 99%.", "url": "https://example.com/eth-layer2"},
    ],
}


def _search(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """Search the mock database using keyword matching."""
    query_lower = query.lower()
    results: list[dict[str, str]] = []

    for topic, entries in _SEARCH_DATABASE.items():
        if topic in query_lower or any(word in query_lower for word in topic.split()):
            results.extend(entries)

    # Fallback: check snippets for partial matches
    if not results:
        query_words = set(query_lower.split())
        for entries in _SEARCH_DATABASE.values():
            for entry in entries:
                snippet_words = set(entry["snippet"].lower().split())
                if query_words & snippet_words:
                    results.append(entry)

    return results[:num_results]


class WebSearchTool(BaseTool):
    """Search the web for current information on a topic (simulated)."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current information on a topic."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        query: str = kwargs.get("query", "")
        num_results: int = kwargs.get("num_results", 5)

        if not query.strip():
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="Query cannot be empty.",
            )

        results = _search(query, num_results)

        if not results:
            return ToolResult(
                tool_name=self.name,
                success=True,
                output={"results": [], "message": f"No results found for '{query}'."},
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            output={"results": results, "total": len(results)},
        )
