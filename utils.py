"""
Utility functions for logging and system checks
"""

import logging
import os
from typing import Dict, Any
from langchain_community.embeddings import OllamaEmbeddings

# Setup logging
logger = logging.getLogger("RAGPlayground")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.FileHandler("rag_playground.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def check_ollama_connection(model_name: str) -> bool:
    """Check if Ollama server is running and model is available"""
    try:
        embeddings = OllamaEmbeddings(model=model_name)
        test_embedding = embeddings.embed_query("test")
        return len(test_embedding) > 0
    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        return False

def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure metadata is JSON-serializable"""
    clean_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            clean_metadata[key] = value
        else:
            clean_metadata[key] = str(value)
    return clean_metadata
