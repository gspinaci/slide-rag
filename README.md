# Slide RAG Chat System

A comprehensive Retrieval-Augmented Generation (RAG) system for PDF slide presentations that enables semantic search and intelligent Q&A using ChromaDB vector database and Large Language Models (LLMs).

## Features

- **PDF Text Extraction**: Extract text from PDF slides with intelligent chunking
- **Vector Database Storage**: Store text chunks in ChromaDB with metadata
- **Semantic Search**: Find relevant content using sentence transformers
- **Multilingual Embeddings**: Support for multilingual models with excellent Italian performance
- **LLM-Powered Answers**: Generate intelligent responses using Gemini or local models
- **Web Chat Interface**: User-friendly Gradio-based chat interface
- **Configurable Chunking**: Adjustable chunk sizes for optimal retrieval performance
- **Multi-Model Support**: Support for both API-based (Gemini) and local LLMs

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Embedding Models](#embedding-models)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [Support](#support)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. ChromaDB Setup

#### Using Docker Compose (Recommended)

The project includes a `docker-compose.yml` file for easy ChromaDB setup:

```yaml
version: '3.8'
services:
  chromadb:
    image: chromadb/chroma
    ports:
      - "8000:8000"
    volumes:
      - ./chroma-data:/data
    restart: unless-stopped
```

## Quick Start

### 1. Start ChromaDB

```bash
docker-compose up -d
```

This starts ChromaDB on `localhost:8000` with persistent storage in `./chroma-data/`.

### 2. Prepare Your PDF Slides

Place your PDF files in the `deck/` folder:

```bash
mkdir -p deck
# Copy your PDF slides to the deck/ folder
```

### 3. Extract and Index PDF Content

```bash
python extract_pdf_text.py
```

This will:
- Extract text from all PDFs in the `deck/` folder
- Create intelligent chunks (default: 800 characters)
- Store chunks in ChromaDB with metadata

### 4. Chat with the slides

You can search and chat with your indexed slides using the command-line search tool:

```bash
python search_slides.py --query "your search query"
```

**Help Output:**
```
Usage: search_slides.py [OPTIONS]

  Semantic search tool for slide content using LangChain and ChromaDB with
  Gemini-powered answers.

Options:
  -q, --query TEXT            Search query string  [required]
  -k, --top-k INTEGER         Number of results to return (default: 5)
  -h, --host TEXT             ChromaDB host (default: localhost)
  -p, --port INTEGER          ChromaDB port (default: 8000)
  -C, --collection TEXT       ChromaDB collection name (default: slide_chunks)
  -m, --model TEXT            Gemini model to use (default: gemini-2.5-flash-
                              lite)
  -E, --embedding_model TEXT  Embedding model to use (default: sentence-
                              transformers/all-MiniLM-L6-v2)
  --help                      Show this message and exit.
```

**What it does:**
The [`search_slides.py`](search_slides.py) script provides a powerful semantic search interface for your indexed slide content. It performs similarity search using vector embeddings to find the most relevant chunks from your slides, then uses Google's Gemini AI model to generate intelligent, contextual answers based on the retrieved content. The tool returns formatted responses with proper references to the source slides and page numbers, making it easy to verify and follow up on the information provided.

**Example Usage:**
```bash
# Basic search
python search_slides.py --query "machine learning algorithms"

# Search with more results
python search_slides.py --query "data preprocessing" --top-k 10

# Use different ChromaDB settings
python search_slides.py --query "neural networks" --host remote-server --port 8001
```


## Configuration

### PDF Extraction Options

The [`extract_pdf_text.py`](extract_pdf_text.py) script supports various configuration options:

```bash
python extract_pdf_text.py --help
```

**Key Parameters:**

- `--folder` (`-f`): Folder containing PDF files (default: `deck`)
- `--collection` (`-c`): ChromaDB collection name (default: `deck_embedings`)
- `--host` (`-h`): ChromaDB host (default: `localhost`)
- `--port` (`-p`): ChromaDB port (default: `8000`)
- `--chunk_size` (`-s`): Maximum characters per chunk (default: `800`)

**Examples:**

```bash
# Extract with custom chunk size
python extract_pdf_text.py --chunk_size 1200

# Use different ChromaDB settings
python extract_pdf_text.py --host remote-server --port 8001

# Process specific folder
python extract_pdf_text.py --folder presentations/
```

### Chunk Size Configuration

The chunk size parameter controls how text is split for vector storage:

- **Small chunks (200-400 chars)**: More precise matching, more chunks per slide
- **Medium chunks (600-800 chars)**: Balanced approach (recommended)
- **Large chunks (1000+ chars)**: More context, fewer chunks

**Chunk Size Impact:**

| Chunk Size | Chunks/Slide | Search Precision | Context Richness |
|------------|--------------|------------------|------------------|
| 200        | ~5           | High             | Low              |
| 800        | ~1-2         | Medium           | High             |
| 1200       | ~1           | Low              | Very High        |

## Embedding Models

### Default vs Multilingual Models

The system uses [`sentence-transformers/all-MiniLM-L6-v2`](search_slides.py:552) as the default embedding model, but supports **multilingual models** that provide significantly better performance for Italian and other non-English content.

## Architecture

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Files     │───▶│  Text Extraction │───▶│   Text Chunks   │
│   (deck/)       │    │  & Chunking      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Chat Interface │◀───│  Semantic Search │◀───│   ChromaDB      │
│  (Gradio)       │    │  & LLM Answer    │    │  Vector Store   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Flow

1. **Extraction**: PDFs → Text extraction → Intelligent chunking
2. **Indexing**: Text chunks → Embeddings → ChromaDB storage
3. **Search**: User query → Semantic search → Relevant chunks
4. **Generation**: Chunks + Query → LLM → Formatted answer

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For issues and questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing issues in the repository
3. Create a new issue with detailed information

---
