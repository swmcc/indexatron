#!/usr/bin/env python3
"""Analyze a single image with LLaVA."""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

from indexatron.analyzer import PhotoAnalyzer

console = Console()


def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage: python analyze_single.py <image_path>[/red]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        console.print(f"[red]Image not found: {image_path}[/red]")
        sys.exit(1)

    # Analyze the image
    analyzer = PhotoAnalyzer()
    result = analyzer.analyze(image_path)

    # Save result
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / f"analysis_{image_path.stem}.json"
    output_data = result.model_dump(mode="json")

    # Remove raw_response for cleaner output file
    display_data = {k: v for k, v in output_data.items() if k != "raw_response"}

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    # Display result
    console.print("\n")
    console.print(Panel(JSON(json.dumps(display_data, indent=2, default=str)), title="📸 Analysis Result"))
    console.print(f"\n[green]✓[/green] Saved to: {output_file}\n")


if __name__ == "__main__":
    main()
