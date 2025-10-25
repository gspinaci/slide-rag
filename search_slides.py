#!/usr/bin/env python3
"""
LangChain-based semantic search system for slide content with LLM-powered answer generation.
Connects to ChromaDB, searches for relevant chunks, and uses Gemini to generate formatted answers.
"""

import sys
import os
from typing import List, Dict, Any
import click
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables from .env file
try:
  from dotenv import load_dotenv
  load_dotenv()  # Load .env file if it exists
except ImportError:
  pass  # dotenv not available, use system environment variables

# Try to import Google Generative AI
try:
  import google.generativeai as genai
  GEMINI_AVAILABLE = True
except ImportError:
  GEMINI_AVAILABLE = False
  genai = None


class LLMAnswerGenerator:
  """
  LLM for generating answers based on search results.
  Supports only Gemini Flash Lite model.
  """

  def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
    """
    Initialize the LLM answer generator.

    Args:
        model_name (str): Gemini model name
    """
    self.model_name = model_name
    self.gemini_model = None
    self._setup_gemini_model()

  def _setup_gemini_model(self):
    """Setup Gemini API model."""
    if not GEMINI_AVAILABLE:
      print("Google Generative AI not available. Install with: pip install google-generativeai")
      self.gemini_model = None
      return

    # Get API key from environment
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
      print(" GOOGLE_API_KEY environment variable not set")
      print("Please set your Google API key: export GOOGLE_API_KEY='your-api-key'")
      self.gemini_model = None
      return

    try:
      print(f"Setting up Gemini model: {self.model_name}")
      genai.configure(api_key=api_key)

      # Use gemini-2.0-flash-exp for gemini-2.5-flash-lite (since it's the available model)
      self.gemini_model = genai.GenerativeModel(self.model_name)

      print(f"Gemini model '{self.model_name}' initialized successfully!")

    except Exception as e:
      print(f" Error setting up Gemini model: {e}")
      self.gemini_model = None

  def generate_answer(self, query: str, search_results: List[Dict[str, Any]], prompt_template: str = None) -> str:
    """
    Generate an answer based on the query and search results.

    Args:
        query (str): The original user query
        search_results (List[Dict]): Search results from ChromaDB
        prompt_template (str): Optional prompt template to use

    Returns:
        str: Generated answer with references
    """
    if not search_results:
      return "No relevant information found in the slides."

    # Use Gemini model
    if self.gemini_model:
      return self._generate_gemini_answer(query, search_results, prompt_template)

  def _generate_gemini_answer(self, query: str, search_results: List[Dict[str, Any]], prompt_template: str = None) -> str:
    """Generate answer using Gemini API."""
    try:
      # Prepare context from search results
      context_parts = []
      references = {}

      for result in search_results:
        deck_name = result['deck_name']
        slide_num = result['slide_number']
        content = result['content']

        context_parts.append(f"From {deck_name}, slide {slide_num}: {content}")

        if deck_name not in references:
          references[deck_name] = []
        if slide_num not in references[deck_name]:
          references[deck_name].append(slide_num)

      context = "\n\n".join(context_parts)

      # Use provided prompt template or default fallback
      if prompt_template:
        prompt = prompt_template.format(context=context, query=query)
      else:
        # Simple fallback prompt
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

      # Generate response with Gemini
      response = self.gemini_model.generate_content(prompt)
      answer = response.text.strip()

      # Format references as a list with bold deck names
      ref_parts = []
      for deck, pages in references.items():
        pages_str = ", ".join(map(str, sorted(pages)))
        ref_parts.append(f"• **{deck}**, pages {pages_str}")

      references_text = "\n".join(ref_parts)

      # Format final answer with references
      formatted_answer = f'{answer}\n\nReferences:\n{references_text}'

      return formatted_answer

    except Exception as e:
      print(f"Error generating Gemini answer: {e}")


class SlideSearcher:
  """
  A semantic search system for slide content using LangChain and ChromaDB with LLM-powered answers.
  """

  def __init__(self, collection_name: str = "slide_chunks", host: str = "localhost", port: int = 8000,
               use_llm: bool = True, llm_model: str = "gemini-2.5-flash-lite", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initialize the slide searcher.

    Args:
        collection_name (str): Name of the ChromaDB collection
        host (str): ChromaDB server host
        port (int): ChromaDB server port
        use_llm (bool): Whether to use LLM for answer generation
        llm_model (str): LLM model name to use
    """
    self.collection_name = collection_name
    self.host = host
    self.port = port
    self.use_llm = use_llm
    self.embeddings = None
    self.vectorstore = None
    self.llm_generator = None
    self.embedding_model = embedding_model

    self._setup_connection()

    self.llm_generator = LLMAnswerGenerator(llm_model)

  def _setup_connection(self):
    """Setup connection to ChromaDB and initialize embeddings."""
    try:
      # Initialize embeddings (same as used in the extraction script)
      print("Loading embeddings model...")
      self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

      # Setup ChromaDB client
      client = chromadb.HttpClient(host=self.host, port=self.port)
      client.heartbeat()
      print(f"Connected to ChromaDB at {self.host}:{self.port}")

      # Initialize LangChain Chroma vectorstore
      self.vectorstore = Chroma(
          client=client,
          collection_name=self.collection_name,
          embedding_function=self.embeddings
      )

      # Check if collection exists and has documents
      collection_info = client.get_collection(self.collection_name)
      doc_count = collection_info.count()
      print(
        f"Connected to collection '{self.collection_name}' with {doc_count} documents")

    except Exception as e:
      print(f"Error connecting to ChromaDB: {e}")
      sys.exit(1)

  def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for the most similar chunks to the query.

    Args:
        query (str): Search query
        k (int): Number of results to return

    Returns:
        List[Dict[str, Any]]: List of search results with metadata
    """
    if not self.vectorstore:
      raise RuntimeError("Vectorstore not initialized")

    try:
      # Perform similarity search with metadata
      results = self.vectorstore.similarity_search_with_score(query, k=k)

      formatted_results = []
      for i, (doc, score) in enumerate(results, 1):
        result = {
            'rank': i,
            'content': doc.page_content,
            'similarity_score': float(score),
            'deck_name': doc.metadata.get('deck_name', 'Unknown'),
            'slide_number': doc.metadata.get('slide_number', 'Unknown'),
            'slide_title': doc.metadata.get('slide_title', ''),
            'chunk_index': doc.metadata.get('chunk_index', 0),
            'source_file': doc.metadata.get('source_file', 'Unknown')
        }
        formatted_results.append(result)

      return formatted_results

    except Exception as e:
      print(f"Error during search: {e}")
      return []

  def generate_answer(self, query: str, k: int = 5, prompt_template: str = None) -> str:
    """
    Search for relevant chunks and generate an LLM-powered answer.

    Args:
        query (str): Search query
        k (int): Number of results to consider for answer generation
        prompt_template (str): Optional prompt template to use

    Returns:
        str: Generated answer with references
    """
    results = self.search(query, k)

    if not results:
      return "No relevant information found in the slides."

    if self.use_llm and self.llm_generator:
      return self.llm_generator.generate_answer(query, results, prompt_template)

  def search_and_answer(self, query: str, k: int = 5, prompt_template: str = None):
    """
    Search for relevant chunks and generate an LLM-powered answer.

    Args:
        query (str): Search query
        k (int): Number of results to consider for answer generation
        prompt_template (str): Optional prompt template to use
    """
    print(f"\nSearching for: '{query}'")
    print("=" * 60)

    print("Generating answer...")
    answer = self.generate_answer(query, k, prompt_template)

    print(f"\nAnswer:")
    print("=" * 60)
    print(answer)
    print("=" * 60)


@click.command()
@click.option('-q', '--query', required=True, help='Search query string')
@click.option('-k', '--top-k', default=5, help='Number of results to return (default: 5)')
@click.option('-h', '--host', default='localhost', help='ChromaDB host (default: localhost)')
@click.option('-p', '--port', default=8000, help='ChromaDB port (default: 8000)')
@click.option('-C', '--collection', default='slide_chunks', help='ChromaDB collection name (default: slide_chunks)')
@click.option('-m', '--model', default='gemini-2.5-flash-lite', help='Gemini model to use (default: gemini-2.5-flash-lite)')
@click.option('-E', '--embedding_model', default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model to use (default: sentence-transformers/all-MiniLM-L6-v2)')
def main(query, top_k, host, port, collection, model, embedding_model):
  """
  Semantic search tool for slide content using LangChain and ChromaDB with Gemini-powered answers.
  """
  print("Slide Semantic Search Tool with Gemini")
  print("=" * 45)

  # Initialize searcher
  searcher = SlideSearcher(
      collection_name=collection,
      host=host,
      port=port,
      llm_model=model,
      embedding_model=embedding_model
  )

  searcher.search_and_answer(query, top_k)


if __name__ == "__main__":
  main()
