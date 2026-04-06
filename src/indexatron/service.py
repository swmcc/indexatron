"""Main service orchestrator for Indexatron."""

import time
from typing import Any

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .analyzer import PhotoAnalyzer
from .api_client import ApiError, McculloghsClient
from .config import get_settings
from .embedder import TextEmbedder
from .logging import (
    console,
    debug,
    debug_config,
    debug_embedding,
    debug_llava,
    error,
    failure,
    info,
    setup_logging,
    success,
    warning,
)


class IndexatronService:
    """Orchestrates the full analysis pipeline."""

    def __init__(self):
        self.settings = get_settings()
        self.api_client = McculloghsClient()
        self.analyzer = PhotoAnalyzer()
        self.embedder = TextEmbedder()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.api_client.close()

    def run(self, limit: int | None = None) -> dict[str, Any]:
        """Run the analysis pipeline.

        Args:
            limit: Maximum number of uploads to process (uses config batch_size if None)

        Returns:
            Summary dict with processed, failed, total_time
        """
        setup_logging()
        debug_config()

        console.print("\n[bold blue]🤖 Indexatron Service[/bold blue]")
        console.print(f"Environment: {self.settings.env}")
        console.print(f"API: {self.settings.api_base_url}\n")

        # Test connections
        if not self._verify_connections():
            return {"processed": 0, "failed": 0, "error": "Connection verification failed"}

        # Fetch pending uploads
        try:
            uploads = self.api_client.fetch_pending_uploads(limit=limit)
        except ApiError as e:
            error(f"Failed to fetch uploads: {e}")
            return {"processed": 0, "failed": 0, "error": str(e)}

        if not uploads:
            console.print("[yellow]No pending uploads to process[/yellow]")
            return {"processed": 0, "failed": 0, "skipped": 0}

        # Process each upload
        results = {"processed": 0, "failed": 0, "total_time": 0.0}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing uploads...", total=len(uploads))

            for upload in uploads:
                short_code = upload["short_code"]
                progress.update(task, description=f"Processing {short_code}...")

                try:
                    start_time = time.time()
                    self._process_upload(upload)
                    elapsed = time.time() - start_time

                    results["processed"] += 1
                    results["total_time"] += elapsed
                    success(f"{short_code} processed in {elapsed:.1f}s")

                except Exception as e:
                    results["failed"] += 1
                    failure(f"{short_code}: {e}")
                    if self.settings.debug:
                        console.print_exception()

                progress.advance(task)

        # Summary
        console.print("\n[bold green]✓ Complete![/bold green]")
        console.print(f"  Processed: {results['processed']}")
        console.print(f"  Failed: {results['failed']}")
        console.print(f"  Total time: {results['total_time']:.1f}s\n")

        return results

    def _verify_connections(self) -> bool:
        """Verify API and Ollama connections."""
        all_ok = True

        # Test API
        info("Testing API connection...")
        if self.api_client.test_connection():
            success("API connection OK")
        else:
            failure("API connection failed")
            all_ok = False

        # Test Ollama
        info("Testing Ollama connection...")
        try:
            import ollama
            ollama.list()
            success("Ollama connection OK")

            # Check for required models
            models = [m.model for m in ollama.list().models]
            vision_model = self.settings.vision_model
            embed_model = self.settings.embedding_model

            if not any(vision_model in m for m in models):
                warning(f"Vision model '{vision_model}' not found - will pull on first use")

            if not any(embed_model in m for m in models):
                warning(f"Embedding model '{embed_model}' not found - will pull on first use")

        except Exception as e:
            failure(f"Ollama connection failed: {e}")
            all_ok = False

        return all_ok

    def _process_upload(self, upload: dict[str, Any]) -> None:
        """Process a single upload: download, analyze, embed, submit.

        Args:
            upload: Upload dict from API
        """
        upload_id = upload["id"]
        short_code = upload["short_code"]

        # Download image
        debug(f"Downloading image for {short_code}...")
        image_path = self.api_client.download_image(upload)

        try:
            # Build metadata context from upload
            metadata = {
                "title": upload.get("title"),
                "caption": upload.get("caption"),
                "date_taken": upload.get("date_taken"),
                "gallery_name": upload.get("gallery_name"),
                "gallery_description": upload.get("gallery_description"),
            }

            # Log metadata if present
            if any(metadata.values()):
                info(f"Using metadata: gallery={metadata.get('gallery_name')}, "
                     f"title={metadata.get('title')}, caption={metadata.get('caption')}, "
                     f"date={metadata.get('date_taken')}")
            else:
                debug("No metadata available for this upload")

            # Analyze with vision model
            debug(f"Analyzing {short_code} with vision model...")
            start = time.time()
            analysis = self.analyzer.analyze(image_path, metadata=metadata)
            elapsed = time.time() - start

            analysis_data = analysis.model_dump(mode="json")
            # Remove raw_response from what we send to API (too large)
            api_analysis = {k: v for k, v in analysis_data.items() if k != "raw_response"}

            debug_llava(
                "Analyze this family photo...",
                analysis_data.get("raw_response", "")[:500],
                elapsed=elapsed,
            )

            # Generate embedding
            debug(f"Generating embedding for {short_code}...")
            embedding_result = self.embedder.embed_analysis(analysis_data, short_code)
            embedding = embedding_result.embedding

            debug_embedding(
                embedding_result.source_text,
                embedding_result.dimensions,
                embedding[:5],
            )

            # Post to API
            debug(f"Posting analysis for {short_code}...")
            self.api_client.post_analysis(upload_id, api_analysis, embedding)

        finally:
            # Cleanup downloaded file
            if image_path.exists():
                image_path.unlink()
                debug(f"Cleaned up {image_path}")

    def process_single(self, upload_id: int) -> bool:
        """Process a single upload by ID.

        Args:
            upload_id: The upload ID to process

        Returns:
            True if successful
        """
        setup_logging()

        # Fetch the specific upload
        uploads = self.api_client.fetch_pending_uploads(limit=100)
        upload = next((u for u in uploads if u["id"] == upload_id), None)

        if not upload:
            error(f"Upload {upload_id} not found or already processed")
            return False

        try:
            self._process_upload(upload)
            success(f"Upload {upload_id} processed successfully")
            return True
        except Exception as e:
            failure(f"Upload {upload_id} failed: {e}")
            if self.settings.debug:
                console.print_exception()
            return False

    def process_by_shortcode(self, shortcode: str) -> bool:
        """Process a single upload by shortcode (for reprocessing).

        Args:
            shortcode: The upload's short_code

        Returns:
            True if successful
        """
        setup_logging()
        debug_config()

        console.print(f"\n[bold blue]🔄 Reprocessing: {shortcode}[/bold blue]\n")

        # Verify connections first
        if not self._verify_connections():
            return False

        # Fetch the specific upload
        upload = self.api_client.fetch_upload_by_shortcode(shortcode)

        if not upload:
            error(f"Upload '{shortcode}' not found")
            return False

        info(f"Found upload: ID={upload['id']}, shortcode={upload['short_code']}")

        try:
            import time
            start_time = time.time()
            self._process_upload(upload)
            elapsed = time.time() - start_time

            success(f"Upload {shortcode} reprocessed in {elapsed:.1f}s")
            return True
        except Exception as e:
            failure(f"Upload {shortcode} failed: {e}")
            if self.settings.debug:
                console.print_exception()
            return False
