"""Reporter for formatting and outputting evaluation results."""

import json
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from evals.core.runner import EvalRunReport


class EvalReporter:
    """Formats and outputs evaluation results."""

    def __init__(self, report: EvalRunReport):
        self.report = report

    def print_summary(self) -> None:
        """Print a human-readable summary to the terminal."""
        if not RICH_AVAILABLE:
            print("\n[Eval Run Summary]")
            print(f"Run ID: {self.report.run_id}")
            print(f"Passed: {self.report.summary['passed']}/{self.report.summary['total_cases']}")
            print(f"Average Score: {self.report.summary['average_score']:.3f}")
            print("\n(Install 'rich' for formatted terminal output)")
            return

        console = Console()
        
        # Main summary panel
        summary = self.report.summary
        total = summary["total_cases"]
        passed = summary["passed"]
        failed = summary["failed"]
        errors = summary["error_count"]
        
        pass_pct = (passed / total * 100) if total > 0 else 0
        fail_pct = (failed / total * 100) if total > 0 else 0
        err_pct = (errors / total * 100) if total > 0 else 0
        
        summary_text = (
            f"Total Cases:     {total}\n"
            f"Passed:          {passed} ({pass_pct:.1f}%)\n"
            f"Failed:          {failed} ({fail_pct:.1f}%)\n"
            f"Errors:          {errors} ({err_pct:.1f}%)\n"
            f"Average Score:   {summary['average_score']:.3f}\n"
            f"Average Steps:   {summary['average_steps']:.1f}"
        )
        
        console.print()
        console.print(Panel(
            summary_text,
            title=f"Eval Run Summary: {self.report.run_id}",
            border_style="blue"
        ))
        
        # Scorer breakdown table (computed on the fly from results)
        scorer_stats: dict[str, dict[str, Any]] = {}
        for r in self.report.results:
            for s in r.scores:
                stats = scorer_stats.setdefault(s.scorer_name, {"total": 0, "passed": 0, "sum": 0.0, "min": 1.0, "max": 0.0})
                stats["total"] += 1
                if s.passed:
                    stats["passed"] += 1
                stats["sum"] += s.score
                stats["min"] = min(stats["min"], s.score)
                stats["max"] = max(stats["max"], s.score)
                
        table = Table(title="Scorer Breakdown")
        table.add_column("Scorer", style="cyan")
        table.add_column("Avg Score", justify="right")
        table.add_column("Pass Rate", justify="right")
        table.add_column("Min/Max", justify="right")
        
        for name, stats in scorer_stats.items():
            avg = stats["sum"] / stats["total"] if stats["total"] > 0 else 0.0
            pr = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
            table.add_row(
                name,
                f"{avg:.3f}",
                f"{pr:.1f}%",
                f"{stats['min']:.2f} / {stats['max']:.2f}"
            )
            
        console.print(table)
        
        # Category breakdown
        cat_table = Table(title="Category Breakdown")
        cat_table.add_column("Category", style="magenta")
        cat_table.add_column("Avg Score", justify="right")
        
        for cat, score in self.report.summary.get("scores_by_category", {}).items():
            cat_table.add_row(cat, f"{score:.3f}")
            
        console.print(cat_table)

        # Failed cases section
        failed_results = [r for r in self.report.results if not r.overall_passed]
        if failed_results:
            console.print("\n[bold red]Failed Cases:[/bold red]")
            for r in failed_results:
                input_trunc = r.case_input[:80] + ("..." if len(r.case_input) > 80 else "")
                
                if r.error:
                    console.print(f"  • [yellow]{r.case_id}[/yellow]: {input_trunc}")
                    console.print(f"    [red]ERROR:[/red] {r.error}")
                else:
                    failed_scorers = [s for s in r.scores if not s.passed]
                    scorer_info = ", ".join([f"{s.scorer_name}({s.score:.2f})" for s in failed_scorers])
                    console.print(f"  • [yellow]{r.case_id}[/yellow]: {input_trunc}")
                    console.print(f"    [red]Failed:[/red] {scorer_info}")
                    
                    # Print reasoning for the first failed scorer
                    if failed_scorers and failed_scorers[0].reasoning:
                        console.print(f"    [dim]Reason: {failed_scorers[0].reasoning}[/dim]")

    def save_json(self, filepath: str) -> None:
        """Save the report as JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.report.to_json())

    def save_markdown(self, filepath: str) -> None:
        """Save the report as Markdown."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.report.summary
        lines = [
            f"# Eval Run Report: {self.report.run_id}",
            f"**Date:** {self.report.timestamp.isoformat()}",
            f"**Agent Provider:** {self.report.agent_info.get('provider')}",
            f"**Agent Model:** {self.report.agent_info.get('model')}",
            "",
            "## Summary",
            f"- **Total Cases:** {summary['total_cases']}",
            f"- **Passed:** {summary['passed']}",
            f"- **Failed:** {summary['failed']}",
            f"- **Errors:** {summary['error_count']}",
            f"- **Average Score:** {summary['average_score']:.3f}",
            "",
            "## Failed Cases",
        ]
        
        failed_results = [r for r in self.report.results if not r.overall_passed]
        if not failed_results:
            lines.append("*None! All cases passed.*")
        else:
            for r in failed_results:
                lines.append(f"### `{r.case_id}`")
                lines.append(f"**Input:** {r.case_input}")
                if r.error:
                    lines.append(f"**Error:** {r.error}")
                else:
                    lines.append("**Failed Scorers:**")
                    for s in r.scores:
                        if not s.passed:
                            lines.append(f"- **{s.scorer_name}** ({s.score:.2f}): {s.reasoning or 'No reasoning'}")
                lines.append("")
                
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def compare(baseline: EvalRunReport, current: EvalRunReport) -> str:
        """Compare two eval runs and return a formatted diff."""
        
        # Simple terminal diff implementation
        lines = [
            f"Comparing: {baseline.run_id} (baseline) -> {current.run_id} (current)",
            "-----------------------------------------------------------"
        ]
        
        b_summary = baseline.summary
        c_summary = current.summary
        
        # Overall diff
        score_diff = c_summary['average_score'] - b_summary['average_score']
        pass_diff = c_summary['passed'] - b_summary['passed']
        
        lines.append(f"Average Score: {b_summary['average_score']:.3f} -> {c_summary['average_score']:.3f} ({score_diff:+.3f})")
        lines.append(f"Passed Cases:  {b_summary['passed']} -> {c_summary['passed']} ({pass_diff:+d})")
        
        # Find regressions and improvements
        b_passed = {r.case_id for r in baseline.results if r.overall_passed}
        c_passed = {r.case_id for r in current.results if r.overall_passed}
        
        regressions = b_passed - c_passed
        improvements = c_passed - b_passed
        
        if regressions:
            lines.append("\nRegressions (Passed -> Failed):")
            for case_id in sorted(regressions):
                lines.append(f"  - {case_id}")
                
        if improvements:
            lines.append("\nImprovements (Failed -> Passed):")
            for case_id in sorted(improvements):
                lines.append(f"  - {case_id}")
                
        return "\n".join(lines)
