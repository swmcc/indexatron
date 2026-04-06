# 🤖 Indexatron

> *Teaching machines to understand family memories*

A Python service that uses local LLMs via Ollama to analyze family photos. Integrates with [the-mcculloughs.org](https://github.com/swmcc/the-mcculloughs.org) to enable semantic search across decades of family memories.

## Features

- **Context-Aware Analysis** - Injects photo metadata (title, caption, date, gallery) into prompts for better results
- **Family Nickname Resolution** - Maps nicknames to real names (e.g., "Mamie" -> "Isobel McCullough")
- **Era Override** - Uses actual `date_taken` instead of AI guessing from visual cues
- **Embeddings** - Generates 768-dimensional vectors with nomic-embed-text for semantic search
- **Safety Filters** - Blocks inappropriate terms, limits category count to prevent repetition loops
- **Reprocessing** - Re-analyse specific photos by shortcode

## Development

| PR | Milestone | Status |
|----|-----------|--------|
| [#5](https://github.com/swmcc/indexatron/pull/5) | Project Setup | ✅ |
| [#1](https://github.com/swmcc/indexatron/pull/1) | Ollama Connection | ✅ |
| [#2](https://github.com/swmcc/indexatron/pull/2) | Image Analysis | ✅ |
| [#3](https://github.com/swmcc/indexatron/pull/3) | Embeddings | ✅ |
| [#4](https://github.com/swmcc/indexatron/pull/4) | Batch Processing | ✅ |
| [#6](https://github.com/swmcc/indexatron/pull/6) | API Integration | ✅ |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                the-mcculloughs.org (Rails)              │
│                                                          │
│  GET /api/uploads/pending      POST /api/uploads/:id/   │
│         │                           analysis             │
│         │                              ▲                 │
└─────────│──────────────────────────────│─────────────────┘
          │                              │
          ▼                              │
┌─────────────────────────────────────────────────────────┐
│                      Indexatron                          │
│                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  Context    │───▶│   LLaVA     │───▶│  Analysis   │ │
│  │  Builder    │    │   (7B)      │    │   JSON      │ │
│  │ (metadata,  │    └─────────────┘    └─────────────┘ │
│  │  aliases)   │           │                            │
│  └─────────────┘           ▼                            │
│                     ┌─────────────┐                     │
│                     │   nomic-    │                     │
│                     │ embed-text  │                     │
│                     └─────────────┘                     │
│                            │                            │
│                            ▼                            │
│                     ┌─────────────┐                     │
│                     │  768-dim    │                     │
│                     │  Embedding  │                     │
│                     └─────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

### Python 3.11+

```bash
python --version  # Should be 3.11 or higher
```

### Ollama

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service (runs in background)
ollama serve

# Or run as a service
brew services start ollama
```

### Required Models

```bash
# Pull the vision model (~4.7GB)
ollama pull llava:7b

# Pull the embedding model (~274MB)
ollama pull nomic-embed-text

# Verify models are installed
ollama list
```

Expected output:
```
NAME                       SIZE
llava:7b                   4.7 GB
nomic-embed-text:latest    274 MB
```

## 🚀 Installation

```bash
# Clone the repo
git clone <repo-url>
cd indexatron

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env.development` or `.env.production` file:

```bash
INDEXATRON_API_KEY=your_api_key
INDEXATRON_API_BASE_URL=http://localhost:3000
INDEXATRON_VISION_MODEL=llava:7b
INDEXATRON_EMBEDDING_MODEL=nomic-embed-text
INDEXATRON_DEBUG=false
```

### Family Nickname Mappings

Edit `src/indexatron/family.py` to add your own nickname mappings:

```python
FAMILY_ALIASES = {
    "mamie": "Isobel McCullough",
    "the oul man": "Edmund McCullough",
    # Add your family nicknames...
}
```

## Project Structure

```
indexatron/
├── README.md
├── pyproject.toml
├── .env.development
├── .env.production
│
└── src/indexatron/
    ├── __init__.py
    ├── cli.py              # CLI entry point
    ├── config.py           # Environment configuration
    ├── service.py          # Main orchestration
    ├── api_client.py       # the-mcculloughs.org API client
    ├── analyzer.py         # LLaVA image analysis
    ├── embedder.py         # Embedding generation
    ├── family.py           # Nickname mappings
    ├── models.py           # Pydantic models
    └── logging.py          # Rich console output
```

## Usage

### Process Pending Uploads

```bash
# Process all pending uploads
indexatron

# Process with a limit
indexatron --limit 10

# Debug mode (verbose output)
indexatron --debug
```

### Reprocess a Specific Photo

```bash
indexatron --shortcode ABC123
```

### Test Connections

```bash
indexatron test
```

### Show Configuration

```bash
indexatron config
```

### Dry Run (fetch without processing)

```bash
indexatron --dry-run
```

## 📊 Output Format

### Analysis JSON

```json
{
  "description": "A family gathering at the beach during sunset...",
  "location": {
    "setting": "beach",
    "type": "outdoor"
  },
  "people": [
    {"description": "young boy, approximately 8 years old", "position": "center"},
    {"description": "woman, approximately 35 years old", "position": "left"}
  ],
  "categories": ["family", "outdoor", "beach", "sunset"],
  "era": {
    "decade": "1990s",
    "confidence": "medium",
    "reasoning": "Photo quality and clothing style suggest mid-90s"
  },
  "mood": "warm, nostalgic, joyful",
  "colors": ["orange", "blue", "golden"],
  "objects": ["beach towel", "sandcastle", "cooler"]
}
```

### Embedding JSON

```json
{
  "filename": "photo.jpg",
  "model": "nomic-embed-text",
  "dimensions": 768,
  "embedding": [0.123, -0.456, 0.789, ...]
}
```

## 🧠 The Models

### LLaVA 7B (Vision)

- **Purpose**: Understand image content
- **Size**: ~4.7GB
- **Strengths**: Good at describing scenes, identifying objects, reading text
- **Limitations**: May hallucinate details, era estimation is approximate

### nomic-embed-text (Embeddings)

- **Purpose**: Convert descriptions to searchable vectors
- **Size**: ~274MB
- **Output**: 768-dimensional vectors
- **Use case**: Finding similar photos via cosine similarity

## Related

- [the-mcculloughs.org](https://github.com/swmcc/the-mcculloughs.org) - Rails family photo app with pgvector
- [Blog: Indexatron](https://swm.cc/writing/indexatron-local-llm-photo-analysis/) - Original experiment write-up
- [Blog: Context-Aware Analysis](https://swm.cc/writing/indexatron-context-aware-analysis/) - How prompt injection improved results

---

*Built with curiosity and local LLMs*
