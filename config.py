"""
Configuration settings for RAG Playground
"""

from typing import Dict, Any

DEFAULT_CONFIG: Dict[str, Any] = {
    "llm_model": "gemma2:2b",
    "embedding_model": "nomic-embed-text",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "retrieval_k": 5,
    "search_type": "similarity",
    "temperature": 0.7,
    "max_tokens": 512,
    "use_hierarchical_retrieval": False,
    "proposition_extraction": False,
    "multimodal_enabled": False,
    "ocr_enabled": False,
    "extract_tables": False,
    "use_reranking": False,
    "use_query_expansion": False,
    "use_hybrid_search": False,
    "enable_pii_redaction": False,
    "storage_path": None,
    "enable_evaluation": False,
    "enable_memory": True,
    "enable_adaptive": False,
    "enable_corrective": False,
    "enable_self_rag": False,
    "enable_hyde": False,
    "enable_branched": False,
    "enable_agentic": False,
    "enable_query_caching": True,
    "enable_query_decomposition": False,
    "use_dynamic_prompting": True,
    "enable_knowledge_graph": False,
    "enable_metadata_filtering": True,
    "use_rrf_fusion": False,
    "enable_clip_embeddings": False,
    "enable_langgraph": False,
    "enable_fine_tuning": False
}

EVALUATION_CONFIG: Dict[str, Any] = {
    "eval_llm_model": "gemma2:2b",
    "embedding_model": "nomic-embed-text",
    "faithfulness_threshold": 0.7,
    "relevance_threshold": 0.7,
    "context_precision_threshold": 0.5,
    "hallucination_severity_threshold": "moderate"
}
