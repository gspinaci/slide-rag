#!/usr/bin/env python3
"""
Minimal script to extract text from PDF slides in the deck/ folder.
This script will be used to prepare text chunks for vector database ingestion.
"""

import os
import sys
import re
from pathlib import Path
import click
from typing import List, Dict, Tuple

try:
  import PyPDF2
  import chromadb
  from chromadb.config import Settings
except ImportError as e:
  print(f"Required package not found: {e}")
  print("Please install dependencies with: pip install -r requirements.txt")
  sys.exit(1)


def extract_text_from_pdf(pdf_path):
  """
  Extract text from a single PDF file.

  Args:
      pdf_path (str): Path to the PDF file

  Returns:
      str: Extracted text from all pages
  """
  text_content = ""

  try:
    with open(pdf_path, 'rb') as file:
      pdf_reader = PyPDF2.PdfReader(file)

      print(f"Processing {pdf_path} - {len(pdf_reader.pages)} pages")

      for page_num, page in enumerate(pdf_reader.pages, 1):
        try:
          page_text = page.extract_text()
          if page_text.strip():  # Only add non-empty pages
            text_content += f"\n--- Page {page_num} ---\n"
            text_content += page_text
            text_content += "\n"
        except Exception as e:
          print(f"Warning: Could not extract text from page {page_num}: {e}")

  except Exception as e:
    print(f"Error processing {pdf_path}: {e}")
    return ""

  return text_content


def process_deck_folder(deck_path="deck"):
  """
  Process all PDF files in the deck folder and extract text.

  Args:
      deck_path (str): Path to the deck folder

  Returns:
      dict: Dictionary with filename as key and extracted text as value
  """
  deck_folder = Path(deck_path)

  if not deck_folder.exists():
    print(f"Error: Deck folder '{deck_path}' not found")
    return {}

  pdf_files = list(deck_folder.glob("*.pdf"))

  if not pdf_files:
    print(f"No PDF files found in '{deck_path}'")
    return {}

  print(f"Found {len(pdf_files)} PDF files")

  extracted_texts = {}

  for pdf_file in sorted(pdf_files):
    print(f"\nProcessing: {pdf_file.name}")
    text = extract_text_from_pdf(pdf_file)

    if text.strip():
      extracted_texts[pdf_file.name] = text
      print(f"Extracted {len(text)} characters from {pdf_file.name}")
    else:
      print(f"✗ No text extracted from {pdf_file.name}")

  return extracted_texts


def save_extracted_text(extracted_texts, output_dir="extracted_text"):
  """
  Save extracted text to individual files for each PDF.

  Args:
      extracted_texts (dict): Dictionary with filename as key and text as value
      output_dir (str): Directory to save the extracted text files
  """
  output_path = Path(output_dir)
  output_path.mkdir(exist_ok=True)

  for filename, text in extracted_texts.items():
    # Create output filename (replace .pdf with .txt)
    output_filename = filename.replace('.pdf', '.txt')
    output_file = output_path / output_filename

    with open(output_file, 'w', encoding='utf-8') as f:
      f.write(text)

    print(f"Saved extracted text to: {output_file}")


def extract_single_deck(deck_input):
  """Extract text from a single PDF deck."""
  # Handle both full paths and just deck names
  if deck_input.startswith('/') or deck_input.startswith('./') or deck_input.startswith('../'):
    # Full path provided
    deck_path = Path(deck_input)
    deck_name = deck_path.stem  # Get filename without extension
  else:
    # Just deck name provided
    deck_name = deck_input
    # Remove .pdf extension if provided
    if deck_name.endswith('.pdf'):
      deck_name = deck_name[:-4]
    deck_path = Path("deck") / f"{deck_name}.pdf"

  if not deck_path.exists():
    print(f"Error: PDF file '{deck_path}' not found")
    return

  print(f"Extracting text from: {deck_path.name}")
  text = extract_text_from_pdf(deck_path)

  if text.strip():
    # Save to individual file
    output_dir = Path("extracted_text")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{deck_name}.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
      f.write(text)

    print(f"Extracted {len(text)} characters")
    print(f"Saved to: {output_file}")
  else:
    print(f"✗ No text extracted from {deck_path.name}")


def text_chunker(text: str, max_chars: int = 250, overlap: int = 50) -> List[str]:
  """
  Split text into chunks of approximately max_chars length with overlap.

  Args:
      text (str): Text to chunk
      max_chars (int): Maximum characters per chunk
      overlap (int): Number of characters to overlap between chunks

  Returns:
      List[str]: List of text chunks
  """
  if not text.strip():
    return []

  # Clean up text - remove excessive whitespace and normalize
  text = re.sub(r'\s+', ' ', text.strip())

  # If text is shorter than max_chars, return as single chunk
  if len(text) <= max_chars:
    return [text]

  chunks = []
  start = 0

  while start < len(text):
    # Find the end position for this chunk
    end = start + max_chars

    if end >= len(text):
      # Last chunk - only add if it has meaningful content
      remaining = text[start:].strip()
      if len(remaining) > 20:  # Only add if substantial content remains
        chunks.append(remaining)
      break

    # Try to break at a sentence boundary
    sentence_end = text.rfind('.', start, end)
    if sentence_end > start + 50:  # Ensure we have substantial content
      end = sentence_end + 1
    else:
      # Try to break at a word boundary
      word_end = text.rfind(' ', start, end)
      if word_end > start + 50:  # Ensure we have substantial content
        end = word_end

    chunk = text[start:end].strip()
    if len(chunk) > 20:  # Only add chunks with meaningful content
      chunks.append(chunk)

    # Move start position with overlap, but ensure we make progress
    next_start = end - overlap
    if next_start <= start:
      next_start = start + max_chars // 2  # Ensure we move forward
    start = next_start

  return chunks


def extract_slide_title(page_text: str) -> str:
  """
  Extract a potential slide title from the page text.

  Args:
      page_text (str): Text from a PDF page

  Returns:
      str: Extracted title or empty string
  """
  lines = page_text.strip().split('\n')

  # Look for the first non-empty line as potential title
  for line in lines:
    line = line.strip()
    if line and len(line) < 100:  # Reasonable title length
      return line

  return ""


def setup_chromadb(host: str = "localhost", port: int = 8000) -> chromadb.Client:
  """
  Setup ChromaDB client connection.

  Args:
      host (str): ChromaDB server host
      port (int): ChromaDB server port

  Returns:
      chromadb.Client: ChromaDB client instance
  """
  try:
    # Simple HttpClient connection
    client = chromadb.HttpClient(host=host, port=port)
    client.heartbeat()
    print(f"Connected to ChromaDB at {host}:{port}")
    return client
  except Exception as e:
    print(f"Error connecting to ChromaDB at {host}:{port}")
    print(f"Make sure ChromaDB server is running: docker-compose up -d")
    print(f"Error: {e}")
    print(f"You can also try running without --chromadb flag to save chunks as text files")
    sys.exit(1)


def store_chunks_in_chromadb(chunks_data: List[Dict], host: str = "localhost", port: int = 8000, collection_name: str = "slide_chunks", deck_name: str = None):
  """
  Store text chunks in ChromaDB with metadata, replacing any existing chunks for the same deck.

  Args:
      chunks_data (List[Dict]): List of chunk data with text and metadata
      collection_name (str): Name of the ChromaDB collection
      deck_name (str): Name of the deck to replace chunks for (if None, uses first chunk's deck_name)
  """
  client = setup_chromadb(host=host, port=port)

  # Get or create collection
  try:
    collection = client.get_collection(collection_name)
    print(f"Using existing collection: {collection_name}")
  except:
    collection = client.create_collection(collection_name)
    print(f"Created new collection: {collection_name}")

  # Determine deck name if not provided
  if deck_name is None and chunks_data:
    deck_name = chunks_data[0]['metadata']['deck_name']

  # Remove existing chunks for this deck
  if deck_name:
    try:
      # Get all documents in the collection
      existing_docs = collection.get()

      # Find IDs that belong to this deck
      ids_to_delete = []
      for doc_id in existing_docs['ids']:
        if doc_id.startswith(f"{deck_name}_"):
          ids_to_delete.append(doc_id)

      # Delete existing chunks for this deck
      if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(
          f"Removed {len(ids_to_delete)} existing chunks")
    except Exception as e:
      print(f"Warning: Could not remove existing chunks: {e}")

  # Prepare data for ChromaDB
  documents = []
  metadatas = []
  ids = []

  for i, chunk_data in enumerate(chunks_data):
    documents.append(chunk_data['text'])
    metadatas.append(chunk_data['metadata'])
    ids.append(
      f"{chunk_data['metadata']['deck_name']}_slide_{chunk_data['metadata']['slide_number']}_chunk_{i}")

  # Add documents to collection
  collection.add(
      documents=documents,
      metadatas=metadatas,
      ids=ids
  )

  print(
    f"Stored {len(chunks_data)} chunks in ChromaDB collection '{collection_name}'")


def process_pdf_with_chunking(pdf_path: Path, chunk_size: int, store_in_db: bool = True) -> List[Dict]:
  """
  Process a PDF file, extract text, and create chunks based on the specified chunk size.

  Args:
      pdf_path (Path): Path to the PDF file
      chunk_size (int): Maximum characters per chunk
      store_in_db (bool): Whether to store chunks in ChromaDB

  Returns:
      List[Dict]: List of chunk data with text and metadata
  """
  deck_name = pdf_path.stem
  chunks_data = []

  try:
    with open(pdf_path, 'rb') as file:
      pdf_reader = PyPDF2.PdfReader(file)

      print(f"Processing {pdf_path.name} - {len(pdf_reader.pages)} pages")

      for page_num, page in enumerate(pdf_reader.pages, 1):
        try:
          page_text = page.extract_text()
          if not page_text.strip():
            continue

          # Extract slide title
          slide_title = extract_slide_title(page_text)

          # Clean up the page text - remove excessive whitespace and normalize
          cleaned_text = re.sub(r'\s+', ' ', page_text.strip())

          # Create chunks from the page text using the chunk_size parameter
          page_chunks = text_chunker(cleaned_text, max_chars=chunk_size, overlap=50)

          # Create chunk data for each chunk from this page
          for chunk_index, chunk_text in enumerate(page_chunks):
            chunk_data = {
                'text': chunk_text,
                'metadata': {
                    'deck_name': deck_name,
                    'slide_title': slide_title,
                    'slide_number': page_num,
                    'chunk_index': chunk_index,
                    'source_file': pdf_path.name
                }
            }
            chunks_data.append(chunk_data)

        except Exception as e:
          print(f"Warning: Could not process page {page_num}: {e}")

  except Exception as e:
    print(f"Error processing {pdf_path}: {e}")
    return []

  print(
    f"Created {len(chunks_data)} chunks from {pdf_path.name}")

  if store_in_db and chunks_data:
    store_chunks_in_chromadb(chunks_data, deck_name=deck_name)

  return chunks_data


@click.command()
@click.option('-f', '--folder', default="deck", help='Folder containing the decks')
@click.option('-c', '--collection', default="deck_embedings", help='Chroma DB collection name')
@click.option('-h', '--host', default="localhost", help='Chroma DB host')
@click.option('-p', '--port', default=8000, help='Chroma DB port')
@click.option('-s', '--chunk_size', default=800, help='The maximum character chunk size')
def main(folder, collection, host, port, chunk_size):
  """PDF Text Extraction Tool for slide decks."""
  print("PDF Text Extraction Tool")
  print("=" * 30)

  all_chunks_data = []

  # Process all decks with chunking
  deck_folder = Path(folder)
  pdf_files = list(deck_folder.glob("*.pdf"))

  if not pdf_files:
    print("No PDF files found in deck folder")
    return

  # Get text for each page
  for pdf_file in sorted(pdf_files):
    chunks_data = process_pdf_with_chunking(pdf_file, chunk_size=chunk_size, store_in_db=False)
    all_chunks_data.extend(chunks_data)

  if all_chunks_data:
    print(f"\nStoring all {len(all_chunks_data)} chunks in ChromaDB...")
    # For "all" mode, we clear the entire collection and replace with new data
    client = setup_chromadb(host=host, port=port)
    try:
      client.delete_collection(collection)
      print(f"Cleared existing collection: {collection}")
    except:
      pass  # Collection might not exist
    store_chunks_in_chromadb(all_chunks_data, host=host, port=port)

  print(f"\nProcessing complete!")
  if chromadb:
    print(f"{len(all_chunks_data)} chunks stored in ChromaDB")
  else:
    print(f"{len(all_chunks_data)} chunks saved to text files")


if __name__ == "__main__":
  main()
