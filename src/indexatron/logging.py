"""Logging configuration with debug support."""

import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from .config import get_settings

# Rich console for output
console = Console()

# Module logger
logger = logging.getLogger("indexatron")


def setup_logging() -> None:
    """Configure logging based on settings."""
    settings = get_settings()

    # Set log level
    level = logging.DEBUG if settings.debug else getattr(logging, settings.log_level)

    # Configure rich handler
    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=settings.debug,
        rich_tracebacks=True,
        tracebacks_show_locals=settings.debug,
    )
    handler.setLevel(level)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[handler],
    )

    # Set our logger level
    logger.setLevel(level)


def debug_request(method: str, url: str, headers: dict | None = None, body: Any = None) -> None:
    """Log HTTP request details in debug mode."""
    if not get_settings().debug:
        return

    parts = [f"[bold cyan]{method}[/bold cyan] {url}"]

    if headers:
        # Mask authorization header
        safe_headers = {
            k: ("sk_****" if k.lower() == "authorization" and v else v)
            for k, v in headers.items()
        }
        parts.append(f"Headers: {safe_headers}")

    if body:
        if isinstance(body, dict) and "embedding" in body:
            # Truncate embedding for display
            display_body = body.copy()
            emb = display_body.get("embedding", [])
            if len(emb) > 5:
                display_body["embedding"] = f"[{len(emb)} floats: {emb[:3]}...]"
            parts.append(f"Body: {display_body}")
        else:
            parts.append(f"Body: {body}")

    console.print(Panel("\n".join(parts), title="[dim]HTTP Request[/dim]", border_style="dim"))


def debug_response(status: int, body: Any = None, elapsed: float | None = None) -> None:
    """Log HTTP response details in debug mode."""
    if not get_settings().debug:
        return

    status_style = "green" if 200 <= status < 300 else "red"
    parts = [f"Status: [bold {status_style}]{status}[/bold {status_style}]"]

    if elapsed:
        parts.append(f"Time: {elapsed:.2f}s")

    if body:
        if isinstance(body, dict):
            # Truncate large responses
            if "uploads" in body:
                count = len(body.get("uploads", []))
                parts.append(f"Uploads: {count} items")
            else:
                parts.append(f"Body: {body}")
        else:
            parts.append(f"Body: {str(body)[:500]}")

    console.print(Panel("\n".join(parts), title="[dim]HTTP Response[/dim]", border_style="dim"))


def debug_llava(prompt: str, response: str, elapsed: float | None = None) -> None:
    """Log LLaVA prompt and response in debug mode."""
    if not get_settings().debug:
        return

    # Show prompt (truncated)
    prompt_display = prompt[:200] + "..." if len(prompt) > 200 else prompt
    console.print(Panel(prompt_display, title="[dim]LLaVA Prompt[/dim]", border_style="blue"))

    # Show response
    response_display = response[:1000] + "..." if len(response) > 1000 else response
    title = "[dim]LLaVA Response[/dim]"
    if elapsed:
        title += f" [dim]({elapsed:.1f}s)[/dim]"
    console.print(Panel(response_display, title=title, border_style="green"))


def debug_embedding(text: str, dimensions: int, preview: list[float]) -> None:
    """Log embedding generation in debug mode."""
    if not get_settings().debug:
        return

    text_display = text[:300] + "..." if len(text) > 300 else text
    preview_str = ", ".join(f"{v:.4f}" for v in preview[:5])

    content = (
        f"Source text: {text_display}\n\n"
        f"Dimensions: {dimensions}\n"
        f"Preview: [{preview_str}, ...]"
    )
    console.print(
        Panel(content, title="[dim]Embedding Generated[/dim]", border_style="magenta")
    )


def debug_config() -> None:
    """Log current configuration in debug mode."""
    if not get_settings().debug:
        return

    settings = get_settings()
    config_lines = [
        f"env: {settings.env}",
        f"debug: {settings.debug}",
        f"api_base_url: {settings.api_base_url}",
        "api_key: sk_****",
        f"ollama_host: {settings.ollama_host}",
        f"vision_model: {settings.vision_model}",
        f"embedding_model: {settings.embedding_model}",
        f"batch_size: {settings.batch_size}",
        f"download_dir: {settings.download_dir}",
    ]
    console.print(
        Panel("\n".join(config_lines), title="[bold]Configuration[/bold]", border_style="yellow")
    )


def info(message: str) -> None:
    """Log info message."""
    logger.info(message)


def debug(message: str) -> None:
    """Log debug message."""
    logger.debug(message)


def warning(message: str) -> None:
    """Log warning message."""
    logger.warning(message)


def error(message: str) -> None:
    """Log error message."""
    logger.error(message)


def success(message: str) -> None:
    """Print success message with checkmark."""
    console.print(f"[green]✓[/green] {message}")


def failure(message: str) -> None:
    """Print failure message with X."""
    console.print(f"[red]✗[/red] {message}")
