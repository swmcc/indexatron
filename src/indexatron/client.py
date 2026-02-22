"""Ollama client wrapper for Indexatron."""

import ollama
from rich.console import Console

console = Console()


class OllamaClient:
    """Wrapper around the Ollama Python client."""

    VISION_MODEL = "llava:7b"
    EMBEDDING_MODEL = "nomic-embed-text"

    def __init__(self):
        self._client = ollama

    def list_models(self) -> list[str]:
        """List available models."""
        response = self._client.list()
        return [model.model for model in response.models]

    def check_required_models(self) -> dict[str, bool]:
        """Check if required models are available."""
        available = self.list_models()
        return {
            "llava:7b": any("llava" in m for m in available),
            "nomic-embed-text": any("nomic-embed-text" in m for m in available),
        }

    def is_ready(self) -> bool:
        """Check if Ollama is ready with required models."""
        status = self.check_required_models()
        return all(status.values())

    def print_status(self):
        """Print connection status to console."""
        console.print("\n[bold blue]🤖 Indexatron - Ollama Connection Test[/bold blue]\n")

        try:
            models = self.list_models()
            console.print(f"[green]✓[/green] Connected to Ollama")
            console.print(f"[green]✓[/green] Found {len(models)} model(s)\n")

            status = self.check_required_models()

            console.print("[bold]Required Models:[/bold]")
            for model, available in status.items():
                if available:
                    console.print(f"  [green]✓[/green] {model}")
                else:
                    console.print(f"  [red]✗[/red] {model} [dim](run: ollama pull {model})[/dim]")

            console.print()
            if self.is_ready():
                console.print("[green bold]Ready to analyze photos![/green bold]\n")
            else:
                console.print("[yellow]Some models are missing. Pull them first.[/yellow]\n")

        except Exception as e:
            console.print(f"[red]✗[/red] Failed to connect to Ollama: {e}")
            console.print("\n[dim]Make sure Ollama is running: ollama serve[/dim]\n")
