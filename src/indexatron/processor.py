"""Batch processing of images."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .analyzer import PhotoAnalyzer
from .embedder import TextEmbedder

console = Console()


class BatchProcessor:
    """Process multiple images and generate combined results."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

    def __init__(self, images_dir: Path, results_dir: Path):
        self.images_dir = images_dir
        self.results_dir = results_dir
        self.analyzer = PhotoAnalyzer()
        self.embedder = TextEmbedder()

    def find_images(self) -> Iterator[Path]:
        """Find all supported images in the images directory."""
        for ext in self.SUPPORTED_EXTENSIONS:
            yield from self.images_dir.glob(f"*{ext}")
            yield from self.images_dir.glob(f"*{ext.upper()}")

    def process_all(self, skip_existing: bool = True) -> dict:
        """Process all images and return combined results."""
        images = list(self.find_images())
        console.print(f"\n[bold blue]🤖 Indexatron Batch Processing[/bold blue]")
        console.print(f"Found {len(images)} images in {self.images_dir}\n")

        results = []
        total_time = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing images...", total=len(images))

            for image_path in images:
                # Check if already processed
                if skip_existing and self._is_processed(image_path):
                    console.print(f"[dim]Skipping (already processed): {image_path.name}[/dim]")
                    progress.advance(task)
                    continue

                progress.update(task, description=f"Processing {image_path.name}...")

                try:
                    start_time = time.time()
                    result = self._process_single(image_path)
                    elapsed = time.time() - start_time
                    total_time += elapsed

                    result["processing_time_seconds"] = round(elapsed, 2)
                    results.append(result)

                    console.print(f"[green]✓[/green] {image_path.name} ({elapsed:.1f}s)")

                except Exception as e:
                    console.print(f"[red]✗[/red] {image_path.name}: {e}")
                    results.append({
                        "filename": image_path.name,
                        "error": str(e),
                    })

                progress.advance(task)

        # Build combined output
        output = {
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "total_images": len(images),
            "processed": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "total_time_seconds": round(total_time, 2),
            "results": results,
        }

        # Save combined results
        output_file = self.results_dir / "batch_results.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2, default=str)

        console.print(f"\n[bold green]✓ Batch processing complete![/bold green]")
        console.print(f"  Processed: {output['processed']}/{output['total_images']}")
        console.print(f"  Total time: {output['total_time_seconds']:.1f}s")
        console.print(f"  Results: {output_file}\n")

        return output

    def _is_processed(self, image_path: Path) -> bool:
        """Check if image has already been processed."""
        analysis_file = self.results_dir / f"analysis_{image_path.stem}.json"
        embedding_file = self.results_dir / f"embedding_{image_path.stem}.json"
        return analysis_file.exists() and embedding_file.exists()

    def _process_single(self, image_path: Path) -> dict:
        """Process a single image: analyze and embed."""
        # Analyze
        analysis = self.analyzer.analyze(image_path)
        analysis_data = analysis.model_dump(mode="json")

        # Save individual analysis
        analysis_file = self.results_dir / f"analysis_{image_path.stem}.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis_data, f, indent=2, default=str)

        # Generate embedding
        embedding = self.embedder.embed_analysis(analysis_data, image_path.name)
        embedding_data = embedding.model_dump(mode="json")

        # Save individual embedding
        embedding_file = self.results_dir / f"embedding_{image_path.stem}.json"
        with open(embedding_file, "w") as f:
            json.dump(embedding_data, f, indent=2, default=str)

        # Return combined result (without raw_response and full embedding for batch file)
        return {
            "filename": image_path.name,
            "analysis": {k: v for k, v in analysis_data.items() if k != "raw_response"},
            "embedding_dimensions": embedding.dimensions,
            "embedding_preview": embedding.embedding[:5],  # First 5 values only
        }
