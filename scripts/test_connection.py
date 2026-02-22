#!/usr/bin/env python3
"""Test connection to Ollama and verify required models."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indexatron.client import OllamaClient


def main():
    client = OllamaClient()
    client.print_status()

    if not client.is_ready():
        sys.exit(1)


if __name__ == "__main__":
    main()
