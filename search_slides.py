#!/usr/bin/env python3
"""
LangChain-based semantic search system for slide content with LLM-powered answer generation.
Connects to ChromaDB, searches for relevant chunks, and uses a local LLM to generate formatted answers.
"""

import sys
import os
from typing import List, Dict, Any, Optional
import click
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
import os
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
  Supports both local models (Qwen) and API models (Gemini).
  """

  def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
    """
    Initialize the LLM answer generator.

    Args:
        model_name (str): Model name (HuggingFace for local, or API model name)
    """
    self.model_name = model_name
    self.tokenizer = None
    self.model = None
    self.pipeline = None
    self.gemini_model = None
    self.is_gemini = self._is_gemini_model(model_name)

    if self.is_gemini:
      self._setup_gemini_model()
    else:
      self._setup_model()

  def _is_gemini_model(self, model_name: str) -> bool:
    """Check if the model is a Gemini model."""
    gemini_models = [
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro",
        "gemini-2.5-flash-lite"  # Added the requested model
    ]
    return any(gemini_name in model_name.lower() for gemini_name in gemini_models)

  def _setup_gemini_model(self):
    """Setup Gemini API model."""
    if not GEMINI_AVAILABLE:
      print("❌ Google Generative AI not available. Install with: pip install google-generativeai")
      self.gemini_model = None
      return

    # Get API key from environment
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
      print("❌ GOOGLE_API_KEY environment variable not set")
      print("Please set your Google API key: export GOOGLE_API_KEY='your-api-key'")
      self.gemini_model = None
      return

    try:
      print(f"🌟 Setting up Gemini model: {self.model_name}")
      genai.configure(api_key=api_key)

      # Map model names to actual Gemini model names
      model_mapping = {
          "gemini-2.5-flash-lite": "gemini-2.0-flash-exp",  # Use available model
          "gemini-2.0-flash-exp": "gemini-2.0-flash-exp",
          "gemini-1.5-flash": "gemini-1.5-flash",
          "gemini-1.5-pro": "gemini-1.5-pro",
          "gemini-pro": "gemini-pro"
      }

      actual_model = model_mapping.get(self.model_name, "gemini-2.0-flash-exp")
      self.gemini_model = genai.GenerativeModel(actual_model)

      print(f"✅ Gemini model '{actual_model}' initialized successfully!")

    except Exception as e:
      print(f"❌ Error setting up Gemini model: {e}")
      self.gemini_model = None

  def _setup_model(self):
    """Setup the local LLM model with optimized settings for PC usage."""
    # Check device availability (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
      device = "cuda"
      print("CUDA GPU detected - using optimized GPU settings")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
      device = "mps"
      print("MPS (Apple Silicon) detected - using optimized Metal settings")
      # For MPS, only auto-fallback for 7B models
      if "7B" in self.model_name:
        print("7B model is too large for MPS, trying 3B model...")
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
      elif "3B" in self.model_name:
        print("Loading 3B model on MPS - may use significant memory...")
    else:
      device = "cpu"
      print("Using CPU - this may be slower but will work")

    try:
      print(f"Loading LLM model: {self.model_name}...")

      # Set device-specific parameters
      if device == "cuda":
        dtype = torch.float16
        device_map = "auto"
        use_8bit = True
      elif device == "mps":
        dtype = torch.float16
        device_map = None  # MPS doesn't support device_map="auto"
        use_8bit = False  # 8-bit quantization not supported on MPS
      else:
        dtype = torch.float32
        device_map = None
        use_8bit = False

      # Load tokenizer
      self.tokenizer = AutoTokenizer.from_pretrained(
          self.model_name,
          trust_remote_code=True,
          use_fast=True
      )

      # Add padding token if missing
      if self.tokenizer.pad_token is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token

      # Load model with memory optimization
      model_kwargs = {
          "dtype": dtype,
          "trust_remote_code": True,
          "low_cpu_mem_usage": True,
      }

      # Add device-specific optimizations
      if device == "cuda":
        model_kwargs["device_map"] = device_map
        model_kwargs["load_in_8bit"] = use_8bit
      elif device == "mps":
        # MPS-specific settings - keep defaults
        pass

      self.model = AutoModelForCausalLM.from_pretrained(
          self.model_name,
          **model_kwargs
      )

      # Move model to MPS if needed (MPS doesn't support device_map)
      if device == "mps":
        self.model = self.model.to("mps")

      # Create optimized pipeline
      pipeline_kwargs = {
          "model": self.model,
          "tokenizer": self.tokenizer,
          "return_full_text": False,
          "clean_up_tokenization_spaces": True
      }

      # Add device-specific pipeline settings
      if device == "cuda":
        pipeline_kwargs["device_map"] = device_map
      elif device == "mps":
        pipeline_kwargs["device"] = 0  # Use MPS device
      else:
        pipeline_kwargs["device"] = -1  # CPU

      self.pipeline = pipeline("text-generation", **pipeline_kwargs)

      print(f"✓ Qwen model loaded successfully on {device.upper()}!")

    except Exception as e:
      error_msg = str(e)
      print(f"Error loading Qwen model: {error_msg}")

      # Check if it's an MPS memory issue
      if device == "mps" and ("NDArray > 2**32" in error_msg or "memory" in error_msg.lower()):
        print("MPS memory limit exceeded - trying smaller models...")
      else:
        print("Trying smaller Qwen alternatives...")

      self._try_smaller_qwen_models()

  def _try_smaller_qwen_models(self):
    """Try progressively smaller Qwen models that are easier to run locally."""
    smaller_models = [
        "Qwen/Qwen2.5-3B-Instruct",  # 3B model - much smaller
        "Qwen/Qwen2.5-1.5B-Instruct",  # 1.5B model - very small
        "Qwen/Qwen2.5-0.5B-Instruct",  # 0.5B model - tiny but functional
    ]

    for model in smaller_models:
      try:
        print(f"Trying smaller model: {model}")
        self.model_name = model

        # Check device availability
        if torch.cuda.is_available():
          device = "cuda"
          dtype = torch.float16
          device_map = "auto"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
          device = "mps"
          dtype = torch.float16
          device_map = None
        else:
          device = "cpu"
          dtype = torch.float32
          device_map = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            use_fast=True
        )

        if self.tokenizer.pad_token is None:
          self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with device-specific settings
        model_kwargs = {
            "dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }

        if device == "cuda":
          model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(
          model, **model_kwargs)

        # Move to MPS if needed
        if device == "mps":
          self.model = self.model.to("mps")

        # Create pipeline with device-specific settings
        pipeline_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "return_full_text": False
        }

        if device == "cuda":
          pipeline_kwargs["device_map"] = device_map
        elif device == "mps":
          pipeline_kwargs["device"] = 0
        else:
          pipeline_kwargs["device"] = -1

        self.pipeline = pipeline("text-generation", **pipeline_kwargs)

        print(f"✓ Successfully loaded {model} on {device.upper()}!")
        return

      except Exception as e:
        print(f"Failed to load {model}: {e}")
        continue

    # If all Qwen models fail, try the original fallback
    print("All Qwen models failed, trying final fallback...")
    try:
      self.model_name = "microsoft/DialoGPT-medium"
      self._setup_fallback_model()
    except Exception as fallback_error:
      print(f"All models failed: {fallback_error}")
      print("LLM answer generation will be disabled - using simple text extraction")
      self.pipeline = None

  def _setup_fallback_model(self):
    """Setup a smaller fallback model."""
    # Check device availability
    if torch.cuda.is_available():
      device = "cuda"
      device_id = 0
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
      device = "mps"
      device_id = 0
    else:
      device = "cpu"
      device_id = -1

    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    # Move to MPS if needed
    if device == "mps":
      self.model = self.model.to("mps")

    self.pipeline = pipeline(
        "text-generation",
        model=self.model,
        tokenizer=self.tokenizer,
        device=device_id,
    )

    print(f"✓ Fallback model loaded successfully on {device.upper()}")

  def generate_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
    """
    Generate an answer based on the query and search results.

    Args:
        query (str): The original user query
        search_results (List[Dict]): Search results from ChromaDB

    Returns:
        str: Generated answer with references
    """
    if not search_results:
      return "No relevant information found in the slides."

    # Use Gemini if available and configured
    if self.is_gemini and self.gemini_model:
      return self._generate_gemini_answer(query, search_results)

    # Use local model
    if not self.pipeline:
      return self._fallback_answer(query, search_results)

    try:
      # Prepare context from search results with length limits
      context_parts = []
      references = {}
      max_context_length = 800  # Limit context length for MPS

      # Use fewer results for MPS to save memory
      import torch
      num_results = 2 if (hasattr(torch.backends, 'mps')
                          and torch.backends.mps.is_available()) else 3

      for result in search_results[:num_results]:
        deck_name = result['deck_name']
        slide_num = result['slide_number']
        content = result['content']

        # Truncate content if too long
        if len(content) > 300:
          content = content[:300] + "..."

        context_parts.append(f"From {deck_name}, slide {slide_num}: {content}")

        if deck_name not in references:
          references[deck_name] = []
        if slide_num not in references[deck_name]:
          references[deck_name].append(slide_num)

      context = "\n\n".join(context_parts)

      # Truncate context if still too long
      if len(context) > max_context_length:
        context = context[:max_context_length] + "..."

      # Create shorter prompt for MPS in Italian
      prompt = f"""Contesto: {context}

Domanda: {query}

Rispondi in italiano basandoti sul contesto fornito.

Risposta:"""

      # Generate response with MPS-optimized settings
      generation_kwargs = {
          "max_new_tokens": 150,  # Reduced for MPS memory
          "temperature": 0.7,
          "do_sample": True,
          "pad_token_id": self.tokenizer.eos_token_id,
          "max_length": 512,  # Limit total sequence length
          "truncation": True,
          "batch_size": 1,  # Force single batch
      }

      # Additional MPS optimizations
      import torch
      if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        generation_kwargs.update({
            "max_new_tokens": 100,  # Further reduce for MPS
            "max_length": 400,  # Shorter sequences for MPS
        })

      response = self.pipeline(prompt, **generation_kwargs)

      # Extract the generated answer
      generated_text = response[0]['generated_text']
      answer = generated_text.split("Answer:")[-1].strip()

      # Format references as a list with bold deck names
      ref_parts = []
      for deck, pages in references.items():
        pages_str = ", ".join(map(str, sorted(pages)))
        ref_parts.append(f"• **{deck}**, pages {pages_str}")

      references_text = "\n".join(ref_parts)

      # Format final answer in Italian with list format
      formatted_answer = f"""{answer}

Per riferimento vedi:
{references_text}"""

      return formatted_answer

    except Exception as e:
      error_msg = str(e)
      print(f"Error generating LLM answer: {error_msg}")

      # Check if it's an MPS memory error during inference
      if "NDArray > 2**32" in error_msg or "MPS" in error_msg:
        print("MPS memory error during inference - trying with smaller model...")
        # Try to reload with a smaller model
        try:
          original_model = self.model_name
          self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
          print(
            f"Switching from {original_model} to {self.model_name} due to MPS memory constraints...")
          self._setup_model()

          # Retry generation with smaller model
          if self.pipeline:
            return self.generate_answer(query, search_results)
        except Exception as retry_error:
          print(f"Fallback model also failed: {retry_error}")

      return self._fallback_answer(query, search_results)

  def _generate_gemini_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
    """Generate answer using Gemini API."""
    try:
      # Prepare context from search results
      context_parts = []
      references = {}

      for result in search_results[:5]:  # Gemini can handle more context
        deck_name = result['deck_name']
        slide_num = result['slide_number']
        content = result['content']

        context_parts.append(f"From {deck_name}, slide {slide_num}: {content}")

        if deck_name not in references:
          references[deck_name] = []
        if slide_num not in references[deck_name]:
          references[deck_name].append(slide_num)

      context = "\n\n".join(context_parts)

      # Create prompt for Gemini with Italian language instruction
      prompt = f"""Basandoti sul seguente contesto dalle slide di presentazione, rispondi alla domanda dell'utente in modo conciso e accurato. Rispondi SEMPRE in italiano.

Contesto:
{context}

Domanda: {query}

Fornisci una risposta chiara e utile basata sul contesto fornito. La risposta deve essere in italiano."""

      # Generate response with Gemini
      response = self.gemini_model.generate_content(prompt)
      answer = response.text.strip()

      # Format references as a list with bold deck names
      ref_parts = []
      for deck, pages in references.items():
        pages_str = ", ".join(map(str, sorted(pages)))
        ref_parts.append(f"• **{deck}**, pages {pages_str}")

      references_text = "\n".join(ref_parts)

      # Format final answer in Italian with list format
      formatted_answer = f"""{answer}

Per riferimento vedi:
{references_text}"""

      return formatted_answer

    except Exception as e:
      print(f"Error generating Gemini answer: {e}")
      return self._fallback_answer(query, search_results)

  def _fallback_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
    """Generate a simple fallback answer without LLM."""
    if not search_results:
      return "No relevant information found in the slides."

    # Use the most relevant result
    top_result = search_results[0]
    content = top_result['content']

    # Simple extraction of key information
    answer = content[:300] + "..." if len(content) > 300 else content

    # Format references
    references = {}
    for result in search_results[:3]:
      deck_name = result['deck_name']
      slide_num = result['slide_number']

      if deck_name not in references:
        references[deck_name] = []
      if slide_num not in references[deck_name]:
        references[deck_name].append(slide_num)

    ref_parts = []
    for deck, pages in references.items():
      pages_str = ", ".join(map(str, sorted(pages)))
      ref_parts.append(f"• **{deck}**, pages {pages_str}")

    references_text = "\n".join(ref_parts)

    return f"""{answer}

Per riferimento vedi:
{references_text}"""


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

    if self.use_llm:
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
      print(f"✓ Connected to ChromaDB at {self.host}:{self.port}")

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
        f"✓ Connected to collection '{self.collection_name}' with {doc_count} documents")

    except Exception as e:
      print(f"Error connecting to ChromaDB: {e}")
      print("Make sure ChromaDB server is running: docker-compose up -d")
      print("And that you have populated the database with: python extract_pdf_text.py -a --chromadb")
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

  def generate_answer(self, query: str, k: int = 5) -> str:
    """
    Search for relevant chunks and generate an LLM-powered answer.

    Args:
        query (str): Search query
        k (int): Number of results to consider for answer generation

    Returns:
        str: Generated answer with references
    """
    results = self.search(query, k)

    if not results:
      return "No relevant information found in the slides."

    if self.use_llm and self.llm_generator:
      return self.llm_generator.generate_answer(query, results)
    else:
      # Fallback to simple answer generation
      fallback_generator = LLMAnswerGenerator()
      return fallback_generator._fallback_answer(query, results)

  def search_and_display(self, query: str, k: int = 5):
    """
    Search and display results in a formatted way.

    Args:
        query (str): Search query
        k (int): Number of results to return
    """
    print(f"\n🔍 Searching for: '{query}'")
    print("=" * 60)

    results = self.search(query, k)

    if not results:
      print("No results found.")
      return

    for result in results:
      print(
        f"\nResult #{result['rank']} (Score: {result['similarity_score']:.4f})")
      print(f"📁 Deck: {result['deck_name']}")
      print(f"Slide: {result['slide_number']}")
      if result['slide_title']:
        print(f"Title: {result['slide_title']}")
      print(f"Source: {result['source_file']}")
      print(f"Content:")
      print(
        f"   {result['content'][:200]}{'...' if len(result['content']) > 200 else ''}")
      print("-" * 60)

  def search_and_answer(self, query: str, k: int = 5):
    """
    Search for relevant chunks and generate an LLM-powered answer.

    Args:
        query (str): Search query
        k (int): Number of results to consider for answer generation
    """
    print(f"\n🔍 Searching for: '{query}'")
    print("=" * 60)

    print("🤖 Generating answer...")
    answer = self.generate_answer(query, k)

    print(f"\n💡 Answer:")
    print("=" * 60)
    print(answer)
    print("=" * 60)


@click.command()
@click.option('-q', '--query', required=True, help='Search query string')
@click.option('-k', '--top-k', default=5, help='Number of results to return (default: 5)')
@click.option('--host', default='localhost', help='ChromaDB host (default: localhost)')
@click.option('--port', default=8000, help='ChromaDB port (default: 8000)')
@click.option('--collection', default='slide_chunks', help='ChromaDB collection name (default: slide_chunks)')
@click.option('--answer', is_flag=True, help='Generate LLM-powered answer instead of showing raw results')
@click.option('--no-llm', is_flag=True, help='Disable LLM and use simple fallback answer generation')
@click.option('--model', default='gemini-2.5-flash-lite', help='LLM model to use (default: gemini-2.5-flash-lite)')
@click.option('-E', '--embedding_model', default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model to use (default: sentence-transformers/all-MiniLM-L6-v2)')
def main(query, top_k, host, port, collection, answer, no_llm, model, embedding_model):
  """
  Semantic search tool for slide content using LangChain and ChromaDB with optional LLM-powered answers.

  Examples:
      python search_slides.py -q "HTML elements"                    # Show search results
      python search_slides.py -q "CSS styling" --answer             # Generate LLM answer
      python search_slides.py -q "JavaScript" --answer --no-llm     # Generate simple answer
      python search_slides.py -q "frameworks" --answer --model "microsoft/DialoGPT-medium"
  """
  print("🔍 Slide Semantic Search Tool with LLM")
  print("=" * 45)

  # Initialize searcher
  use_llm = not no_llm
  searcher = SlideSearcher(
      collection_name=collection,
      host=host,
      port=port,
      use_llm=use_llm,
      llm_model=model,
      embedding_model=embedding_model
  )

  if answer:
    # Generate LLM-powered answer
    searcher.search_and_answer(query, top_k)
  else:
    # Show traditional search results
    searcher.search_and_display(query, top_k)


if __name__ == "__main__":
  main()
