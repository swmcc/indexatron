#!/usr/bin/env python3
"""Process all images in test_images directory."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indexatron.processor import BatchProcessor


def main():
    project_root = Path(__file__).parent.parent
    images_dir = project_root / "test_images"
    results_dir = project_root / "results"

    # Ensure results directory exists
    results_dir.mkdir(exist_ok=True)

    # Process all images
    processor = BatchProcessor(images_dir, results_dir)

    # Use --force to reprocess existing images
    skip_existing = "--force" not in sys.argv

    processor.process_all(skip_existing=skip_existing)


if __name__ == "__main__":
    main()
