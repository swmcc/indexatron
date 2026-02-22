"""Embedding generation using nomic-embed-text."""

import ollama
from rich.console import Console

from .models import EmbeddingResult

console = Console()


class TextEmbedder:
    """Generates embeddings using nomic-embed-text."""

    MODEL = "nomic-embed-text"
    DIMENSIONS = 768

    def embed(self, text: str, filename: str = "unknown") -> EmbeddingResult:
        """Generate embedding for text."""
        console.print(f"\n[bold blue]🧮 Generating embedding for:[/bold blue] {filename}")
        console.print(f"[dim]Text length: {len(text)} chars[/dim]")

        response = ollama.embed(model=self.MODEL, input=text)

        # The response contains embeddings as a list (we only sent one text)
        embedding = response.embeddings[0]

        console.print(f"[green]✓[/green] Generated {len(embedding)}-dimensional embedding")

        return EmbeddingResult(
            filename=filename,
            model_used=self.MODEL,
            dimensions=len(embedding),
            embedding=embedding,
            source_text=text[:500] + "..." if len(text) > 500 else text,
        )

    def embed_analysis(self, analysis_json: dict, filename: str) -> EmbeddingResult:
        """Generate embedding from a photo analysis result."""
        # Create a text representation of the analysis for embedding
        parts = []

        if desc := analysis_json.get("description"):
            parts.append(desc)

        if location := analysis_json.get("location"):
            if isinstance(location, dict):
                parts.append(f"Location: {location.get('setting', '')} {location.get('type', '')}")

        if categories := analysis_json.get("categories"):
            if isinstance(categories, list):
                parts.append(f"Categories: {', '.join(categories)}")

        if mood := analysis_json.get("mood"):
            parts.append(f"Mood: {mood}")

        if era := analysis_json.get("era"):
            if isinstance(era, dict):
                parts.append(f"Era: {era.get('decade', 'unknown')}")

        if objects := analysis_json.get("objects"):
            if isinstance(objects, list):
                parts.append(f"Objects: {', '.join(str(o) for o in objects)}")

        text = " | ".join(parts)
        return self.embed(text, filename)
