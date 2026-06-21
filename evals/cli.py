"""CLI entry point for running evaluations."""

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys

import yaml

from evals.adapters.example_agent import ExampleAgentAdapter
from evals.core.dataset import EvalDataset
from evals.core.reporter import EvalReporter
from evals.core.runner import EvalConfig, EvalRunner, EvalRunReport

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> EvalConfig:
    """Load config from YAML."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return EvalConfig()
        
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    return EvalConfig.model_validate(data)


async def run_cmd(args: argparse.Namespace) -> int:
    """Execute the 'run' command."""
    # 1. Load config
    config = load_config(args.config)
    
    # Override from CLI
    if args.tags:
        config.tags_filter = [t.strip() for t in args.tags.split(",")]
    if args.category:
        config.category_filter = args.category
    if args.output:
        config.output_dir = args.output
        
    # 2. Load dataset
    try:
        dataset = EvalDataset(args.dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1
        
    # 3. Instantiate adapter
    # In a real framework with multiple adapters, we'd use a registry or factory
    if args.adapter == "example":
        # We rely on env vars or .env file for API keys
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        provider = os.getenv("AGENT_LLM_PROVIDER", "openai")
        model = os.getenv("AGENT_MODEL_NAME", "gpt-4o-mini")
        api_key = os.getenv("AGENT_API_KEY", "")
        
        if not api_key:
            logger.warning("No AGENT_API_KEY found in environment. Example agent may fail.")
            
        adapter = ExampleAgentAdapter(
            provider=provider,
            model=model,
            api_key=api_key,
        )
    else:
        logger.error(f"Unknown adapter: {args.adapter}")
        return 1
        
    # 3b. Resolve LLM judge API key from environment if configured
    if config.llm_judge_config and "api_key_env" in config.llm_judge_config:
        env_var = config.llm_judge_config.pop("api_key_env")
        config.llm_judge_config["api_key"] = os.getenv(env_var, "")
        if not config.llm_judge_config["api_key"]:
            logger.warning(f"No {env_var} found in environment. LLM judge may fail.")

    # 4. Run evals
    runner = EvalRunner(adapter, dataset, config)
    report = await runner.run()
    
    # 5. Output results
    reporter = EvalReporter(report)
    
    if args.format in ("terminal", "all"):
        reporter.print_summary()
        
    if args.format in ("markdown", "all"):
        md_path = Path(config.output_dir) / f"{report.run_id}.md"
        reporter.save_markdown(str(md_path))
        logger.info(f"Markdown report saved to {md_path}")
        
    # JSON is always saved by the runner, but we can explicitly log it
    logger.info(f"JSON report saved to {config.output_dir}/{report.run_id}.json")
    
    # Exit with 1 if there were failures
    return 0 if report.summary["failed"] == 0 and report.summary["error_count"] == 0 else 1


def dataset_info_cmd(args: argparse.Namespace) -> int:
    """Execute the 'dataset-info' command."""
    try:
        dataset = EvalDataset(args.path)
        summary = dataset.summary()
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1


def compare_cmd(args: argparse.Namespace) -> int:
    """Execute the 'compare' command."""
    try:
        with open(args.baseline, "r") as f:
            b_data = json.load(f)
            b_report = EvalRunReport.model_validate(b_data)
            
        with open(args.current, "r") as f:
            c_data = json.load(f)
            c_report = EvalRunReport.model_validate(c_data)
            
        diff_text = EvalReporter.compare(b_report, c_report)
        print("\n" + diff_text + "\n")
        return 0
    except Exception as e:
        logger.error(f"Failed to compare reports: {e}")
        return 1


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evals Framework CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # 'run' command
    run_parser = subparsers.add_parser("run", help="Execute an eval run")
    run_parser.add_argument("--config", default="evals/configs/default.yaml", help="Path to config file")
    run_parser.add_argument("--dataset", default="evals/datasets", help="Path to dataset directory or file")
    run_parser.add_argument("--adapter", default="example", help="Which agent adapter to use")
    run_parser.add_argument("--tags", help="Comma-separated tags to filter by")
    run_parser.add_argument("--category", help="Category to filter by")
    run_parser.add_argument("--output", help="Override output directory")
    run_parser.add_argument("--format", choices=["terminal", "json", "markdown", "all"], default="terminal")
    
    # 'dataset-info' command
    ds_parser = subparsers.add_parser("dataset-info", help="Print dataset statistics")
    ds_parser.add_argument("--path", required=True, help="Path to dataset file or directory")
    
    # 'compare' command
    cmp_parser = subparsers.add_parser("compare", help="Compare two eval runs")
    cmp_parser.add_argument("--baseline", required=True, help="Path to baseline JSON report")
    cmp_parser.add_argument("--current", required=True, help="Path to current JSON report")
    cmp_parser.add_argument("--format", default="terminal")
    
    args = parser.parse_args()
    
    if args.command == "run":
        # Disable logging from the underlying HTTP clients
        logging.getLogger("httpx").setLevel(logging.WARNING)
        sys.exit(asyncio.run(run_cmd(args)))
    elif args.command == "dataset-info":
        sys.exit(dataset_info_cmd(args))
    elif args.command == "compare":
        sys.exit(compare_cmd(args))


if __name__ == "__main__":
    main()
