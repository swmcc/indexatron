# 🤖 Indexatron

> *Teaching machines to understand family memories*

An experimental Python service that uses local LLMs via Ollama to analyze family photos. Part of a larger experiment to enable semantic search across decades of family memories.

## 🧪 Experiment Status

This is a **science experiment** - proving that local AI can meaningfully analyze family photos before integrating with a production system.

| Branch | Status | What it proves |
|--------|--------|----------------|
| `01-project-setup` | ✅ | Project structure works |
| `02-ollama-connection` | ✅ | Can talk to Ollama |
| `03-image-analysis` | ✅ | LLaVA understands photos |
| `04-embedding-generation` | ✅ | Can generate embeddings |
| `05-batch-processing` | ✅ | Can process many photos |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Indexatron                          │
│                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   Image     │───▶│   LLaVA     │───▶│  Analysis   │ │
│  │   Input     │    │   (7B)      │    │   JSON      │ │
│  └─────────────┘    └─────────────┘    └─────────────┘ │
│                            │                            │
│                            ▼                            │
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
                            │
                            ▼
                    ┌─────────────────┐
                    │  Results JSON   │
                    │  (for now)      │
                    └─────────────────┘
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

## 📁 Project Structure

```
indexatron/
├── README.md                 # You are here
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Modern packaging
├── .gitignore
│
├── src/indexatron/          # Main package
│   ├── __init__.py
│   ├── client.py            # Ollama client wrapper
│   ├── analyzer.py          # LLaVA image analysis
│   ├── embedder.py          # Embedding generation
│   ├── processor.py         # Batch processing
│   └── models.py            # Pydantic models
│
├── scripts/                 # CLI scripts
│   ├── test_connection.py   # Verify Ollama works
│   ├── analyze_single.py    # Analyze one image
│   ├── generate_embedding.py
│   └── process_batch.py     # Process all images
│
├── test_images/             # Sample images (git tracked)
│   └── ...
│
└── results/                 # Output (gitignored)
    └── ...
```

## 🔬 Usage

### Test Ollama Connection

```bash
python scripts/test_connection.py
```

### Analyze a Single Image

```bash
python scripts/analyze_single.py test_images/photo.jpg
# Output: results/analysis_photo.json
```

### Generate Embedding

```bash
python scripts/generate_embedding.py test_images/photo.jpg
# Output: results/embedding_photo.json
```

### Batch Process All Images

```bash
python scripts/process_batch.py
# Output: results/batch_results.json
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

## 🔗 Related

This is part of a larger experiment:

- **Rails API**: Provides photo storage and similarity search endpoints
- **Indexatron** (this): Analyzes photos and generates embeddings
- **Future**: UI for browsing results and finding similar photos

## 📝 Experiment Log

### 2026-02-22

- Initial project setup
- Testing LLaVA on family photos
- Generating first embeddings

---

*🤖 Built with curiosity and local LLMs*
