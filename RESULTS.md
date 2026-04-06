# Indexatron: Results

> **Status:** ✅ SUCCESS
> **Date:** 2026-02-22
> **Hypothesis:** Local LLMs can analyze family photos with useful metadata extraction

> **Status:** ✅ IN PRODUCTION
> **Date:** 2026-04-06
> **Hypothesis:** Context-aware prompting dramatically improves results

## Executive Summary

The experiment was a success and is now in production. Ollama running locally with LLaVA:7b and nomic-embed-text can:

1. **Analyze photos** - Extract descriptions, detect people/objects, estimate era
2. **Generate embeddings** - Create 768-dimensional vectors for similarity search
3. **Process batches** - Handle multiple images with progress tracking
4. **Use context** - Inject metadata for dramatically better results

The service now integrates with [the-mcculloughs.org](https://github.com/swmcc/the-mcculloughs.org) for automated photo analysis.

## Test Results

| Metric | Value |
|--------|-------|
| Images Processed | 3/3 |
| Failed | 0 |
| Total Time | 40.82s |
| Avg Time/Image | ~13.6s |

### Sample Outputs

#### 🐕 family_photo_03.jpg
- **Description:** "A tan-colored Labrador Retriever is sitting on a wooden floor indoors"
- **Categories:** `["dog"]`
- **Mood:** calm
- **Processing Time:** 14.73s

#### 🍺 family_photo_02.jpg
- **Description:** "A photo of a bottle of beer and a glass with frothy white head on top, placed on a table at a restaurant"
- **Location:** Indoor restaurant
- **Objects Detected:** Beer bottle (Kingfisher brand), glass with beer
- **Categories:** `["beer", "restaurant"]`
- **Processing Time:** 14.2s

#### 👔 family_photo_01.jpg
- **Description:** "A man standing in an indoor conference room during a wedding reception"
- **Era Detected:** 2010s (medium confidence)
- **Person:** Male guest, 30s, wearing suit and tie
- **Categories:** `["wedding"]`
- **Processing Time:** 11.89s

## What Worked Well

### LLaVA Vision Analysis
- Correctly identified subjects (dog, beer, person)
- Detected specific brands (Kingfisher)
- Estimated era from visual cues
- Provided useful mood/atmosphere descriptions

### Embedding Generation
- 768-dimensional embeddings generated for all images
- Based on analysis descriptions (semantic meaning)
- Ready for similarity search when needed

### Batch Processing
- Progress bar with Rich library
- Skip existing functionality
- Combined JSON output

## Model Comparison: LLaVA 7b vs Llama 3.2 Vision

Tested both models extensively. Llama 3.2 Vision is newer and larger (7.8GB vs 4.7GB), but LLaVA proved more reliable for this use case.

| Aspect | LLaVA 7b | Llama 3.2 Vision |
|--------|----------|------------------|
| Speed | ~27s per image | ~60s+ per image |
| JSON Output | Mostly valid, occasional truncation | More verbose, sometimes malformed |
| Structured Output | Follows schema reliably | Occasionally enters repetition loops |
| Context Adherence | Good with explicit prompts | Variable, may need different prompting |

**Conclusion:** LLaVA 7b's tighter, more predictable outputs made it better suited for this structured extraction pipeline. Llama 3.2 Vision might shine in conversational tasks, but for constrained schema-driven work the smaller model proved more reliable.

## Context-Aware Analysis

The breakthrough was injecting domain knowledge into prompts. Instead of asking "what's in this photo?", we tell the AI what it should already know:

```
IMPORTANT: This photo includes Edmund McCullough, Isobel McCullough.
Use these REAL names in the 'people' array.

IMPORTANT: This photo is from 1974-08-14 (1970s).
Use this as the era decade with 'high' confidence.

This photo is from the album: "Old 35mm Slides"
Caption says: "Auntie Wilma, Mum, Dad and Uncle Sam"
```

This transforms the AI from a generic image analyser into something that understands your specific photos.

### Family Nickname Resolution

The system scans titles, captions, and gallery names for known aliases and injects the real names into the prompt:

- "Mamie" -> Isobel McCullough
- "The Oul Man" -> Edmund McCullough
- "The Bro" -> John McCullough

### Era Override

Rather than trust the AI's visual guesses, we override with actual `date_taken` when available. A photo from 1974 is correctly tagged as 1970s with high confidence.

## Quirks & Learnings

### JSON Parsing Required Repair
LLaVA doesn't always output clean JSON. The analyzer needed:
- Code block stripping
- Brace balancing
- Type coercion for nested objects

### Model Hallucinations
Some amusing observations:
- The dog photo mentioned "clothing" and "fashion trends for pets" (the dog had no clothes)
- Beer was classified under `people` array with `estimated_age: "Beer is an alcoholic beverage"`

These quirks don't break the system. Robust parsing handles them.

### Llama 3.2 Vision Repetition Loops
The newer model occasionally produces output like:
```json
{"categories": ["family", "children", "family", "children", "family"...]}
```

Safety filters now cap category arrays at 20 items to prevent this flooding the database.

### Inappropriate Content Filtering
Vision models can sometimes generate unsuitable descriptions for family photos. A blocklist filters these terms before storing.

### Processing Time
~27 seconds per image with LLaVA on CPU. Optimisations that helped:
- Resize images to 1024px max before analysis
- Use pre-generated medium variants from Rails
- Convert WebP to JPG (LLaVA segfaults on WebP)

## Development Progress

Each milestone was developed incrementally:

| PR | Milestone | What It Proved |
|----|-----------|----------------|
| [#5](https://github.com/swmcc/indexatron/pull/5) | Project Setup | Foundation ready |
| [#1](https://github.com/swmcc/indexatron/pull/1) | Ollama Connection | Local LLM runtime accessible |
| [#2](https://github.com/swmcc/indexatron/pull/2) | Image Analysis | LLaVA extracts useful metadata |
| [#3](https://github.com/swmcc/indexatron/pull/3) | Embeddings | 768-dim vectors for similarity |
| [#4](https://github.com/swmcc/indexatron/pull/4) | Batch Processing | Scalable to many images |
| [#6](https://github.com/swmcc/indexatron/pull/6) | API Integration | Production service for the-mcculloughs.org |

## Technical Stack

```
Ollama (local runtime)
├── llava:7b (~4.7GB) - Vision analysis
└── nomic-embed-text (~274MB) - Embeddings

Python 3.11+
├── ollama - API client
├── pydantic - Data validation
├── pillow - Image handling
└── rich - Console output
```

## Next Steps

Completed:
- ✅ Rails API integration
- ✅ Database storage with pgvector
- ✅ Context-aware analysis

Future:
- [ ] [Search API](https://github.com/swmcc/the-mcculloughs.org/issues/99) - Query by person, category, decade
- [ ] Semantic search - Find photos similar to a given photo using embeddings
- [ ] Face recognition - Cluster photos by person

## Conclusion

Local LLMs provide a privacy-preserving alternative to cloud APIs for photo analysis. The quality is good enough for family photo organization, and context injection transforms generic analysis into something that understands your specific archive.

---

*Built with curiosity and local LLMs*
