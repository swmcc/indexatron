#!/usr/bin/env python3
"""Generate embedding for an image (requires analysis first)."""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

from indexatron.embedder import TextEmbedder

console = Console()


def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage: python generate_embedding.py <image_path>[/red]")
        console.print("[dim]Note: Run analyze_single.py first to create the analysis file[/dim]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    results_dir = Path(__file__).parent.parent / "results"

    # Look for existing analysis
    analysis_file = results_dir / f"analysis_{image_path.stem}.json"
    if not analysis_file.exists():
        console.print(f"[red]Analysis not found: {analysis_file}[/red]")
        console.print("[dim]Run: python scripts/analyze_single.py {image_path}[/dim]")
        sys.exit(1)

    # Load analysis
    with open(analysis_file) as f:
        analysis_data = json.load(f)

    # Generate embedding
    embedder = TextEmbedder()
    result = embedder.embed_analysis(analysis_data, image_path.name)

    # Save embedding
    output_file = results_dir / f"embedding_{image_path.stem}.json"
    output_data = result.model_dump(mode="json")

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    # Display summary
    console.print(f"\n[green]✓[/green] Saved embedding to: {output_file}")
    console.print(f"[dim]Dimensions: {result.dimensions}[/dim]")
    console.print(f"[dim]Source text: {result.source_text[:100]}...[/dim]\n")


if __name__ == "__main__":
    main()
