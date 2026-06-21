"""Interactive CLI for the agent — useful for manual testing."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.app import Agent, AgentTrace
from agent.config import Settings

console = Console()


def _print_trace(trace: AgentTrace) -> None:
    """Pretty-print the agent trace using Rich."""
    # Steps table
    table = Table(
        title="Execution Trace",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Action", style="cyan", width=14)
    table.add_column("Tool", style="green", width=22)
    table.add_column("Reasoning", style="white", max_width=60)

    for step in trace.steps:
        tool_info = step.tool_name or "—"
        if step.tool_args:
            args_str = json.dumps(step.tool_args, ensure_ascii=False)
            if len(args_str) > 40:
                args_str = args_str[:37] + "..."
            tool_info = f"{tool_info}({args_str})"

        table.add_row(
            str(step.step_number),
            step.action,
            tool_info,
            step.reasoning[:60] + "..." if len(step.reasoning) > 60 else step.reasoning,
        )

    console.print(table)

    # Summary
    summary = Text()
    summary.append(f"Steps: {trace.total_steps}", style="bold")
    summary.append(f"  |  Latency: {trace.total_latency_ms:.0f}ms", style="dim")
    if trace.safety_triggered:
        summary.append("  |  ⚠️ Safety triggered", style="bold red")
    if trace.error:
        summary.append(f"  |  ❌ Error: {trace.error}", style="bold red")
    console.print(summary)


async def _run_repl(agent: Agent, verbose: bool) -> None:
    """Run the interactive REPL loop."""
    console.print(
        Panel(
            "[bold cyan]Research Assistant Agent[/bold cyan]\n"
            "Type your question and press Enter. Type 'quit' or 'exit' to leave.\n"
            "Tools available: web_search, calculator, get_weather, knowledge_base_search",
            title="🤖 Agent CLI",
            border_style="bright_blue",
        )
    )

    while True:
        try:
            user_input = console.input("\n[bold green]You>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
            trace = await agent.run(user_input)

        # Print response
        console.print(
            Panel(
                trace.output,
                title="🤖 Assistant",
                border_style="bright_cyan",
                padding=(1, 2),
            )
        )

        if verbose:
            _print_trace(trace)


def main() -> None:
    """Entry point for the agent CLI."""
    parser = argparse.ArgumentParser(
        description="Interactive CLI for the Research Assistant Agent",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print the full execution trace for each response.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider (openai, anthropic, gemini). Overrides env var.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. Overrides env var.",
    )
    args = parser.parse_args()

    # Load config
    settings = Settings()

    if args.provider:
        settings.llm_provider = args.provider
    if args.model:
        settings.model_name = args.model

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not settings.api_key:
        console.print(
            "[bold red]Error:[/bold red] No API key found. "
            "Set the AGENT_API_KEY environment variable or add it to .env",
        )
        sys.exit(1)

    # Create agent
    agent = Agent(
        provider=settings.llm_provider,
        model=settings.model_name,
        api_key=settings.api_key,
        max_steps=settings.max_steps,
        temperature=settings.temperature,
    )

    # Run
    asyncio.run(_run_repl(agent, verbose=args.verbose))


if __name__ == "__main__":
    main()
