"""Command-line interface for Indexatron."""

import argparse
import os
import sys

from rich.console import Console

console = Console()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="indexatron",
        description="AI-powered family photo analysis using local LLMs",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (verbose logging)",
    )

    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Maximum number of uploads to process",
    )

    parser.add_argument(
        "--env",
        choices=["development", "production"],
        default=None,
        help="Environment to use (overrides INDEXATRON_ENV)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch uploads but don't process them",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command (default)
    run_parser = subparsers.add_parser("run", help="Process pending uploads")
    run_parser.add_argument("--limit", "-n", type=int, help="Max uploads to process")

    # test command
    subparsers.add_parser("test", help="Test connections to API and Ollama")

    # config command
    subparsers.add_parser("config", help="Show current configuration")

    args = parser.parse_args()

    # Set environment variables before importing config
    if args.env:
        os.environ["INDEXATRON_ENV"] = args.env

    if args.debug:
        os.environ["INDEXATRON_DEBUG"] = "true"

    # Import after env vars are set
    from .config import get_settings, reload_settings

    # Reload settings if we changed env vars
    if args.env or args.debug:
        reload_settings()

    settings = get_settings()

    # Handle commands
    if args.command == "test":
        _cmd_test(settings)
    elif args.command == "config":
        _cmd_config(settings)
    elif args.dry_run:
        _cmd_dry_run(settings, args.limit)
    else:
        _cmd_run(settings, args.limit)


def _cmd_run(settings, limit: int | None):
    """Run the analysis pipeline."""
    from .service import IndexatronService

    with IndexatronService() as service:
        results = service.run(limit=limit)

    # Exit with error code if all failed
    if results.get("processed", 0) == 0 and results.get("failed", 0) > 0:
        sys.exit(1)


def _cmd_test(settings):
    """Test connections."""
    from .api_client import McculloghsClient
    from .logging import failure, setup_logging, success

    setup_logging()
    console.print("\n[bold]Testing Connections[/bold]\n")

    all_ok = True

    # Test API
    console.print("API connection...", end=" ")
    try:
        with McculloghsClient() as client:
            if client.test_connection():
                success("OK")
            else:
                failure("Failed (check API key)")
                all_ok = False
    except Exception as e:
        failure(f"Error: {e}")
        all_ok = False

    # Test Ollama
    console.print("Ollama connection...", end=" ")
    try:
        import ollama
        ollama.list()
        success("OK")

        # Check models
        models = [m.model for m in ollama.list().models]
        console.print(f"  Available models: {', '.join(models[:5])}")

        if not any(settings.vision_model in m for m in models):
            console.print(f"  [yellow]⚠ Vision model '{settings.vision_model}' not found[/yellow]")

        if not any(settings.embedding_model in m for m in models):
            console.print(
                f"  [yellow]⚠ Embedding model '{settings.embedding_model}' not found[/yellow]"
            )

    except Exception as e:
        failure(f"Error: {e}")
        all_ok = False

    console.print()
    if all_ok:
        console.print("[bold green]All connections OK[/bold green]\n")
    else:
        console.print("[bold red]Some connections failed[/bold red]\n")
        sys.exit(1)


def _cmd_config(settings):
    """Show current configuration."""
    from .logging import debug_config, setup_logging

    setup_logging()

    # Force debug mode temporarily to show config
    original_debug = settings.debug
    settings.__dict__["debug"] = True
    debug_config()
    settings.__dict__["debug"] = original_debug


def _cmd_dry_run(settings, limit: int | None):
    """Fetch uploads but don't process."""
    from .api_client import McculloghsClient
    from .logging import setup_logging

    setup_logging()
    console.print("\n[bold]Dry Run Mode[/bold]\n")

    with McculloghsClient() as client:
        uploads = client.fetch_pending_uploads(limit=limit)

    if not uploads:
        console.print("[yellow]No pending uploads[/yellow]\n")
        return

    console.print(f"Found {len(uploads)} pending uploads:\n")
    for u in uploads:
        console.print(f"  • {u['short_code']} (ID: {u['id']}) - {u['created_at']}")

    console.print()


if __name__ == "__main__":
    main()
