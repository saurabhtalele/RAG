from typing import Dict, Any

DEFAULT_CONFIG: Dict[str, Any] = {
    # 🤖 LLM Settings
    "llm_model": "gemma2:2b",
    "temperature": 0.7,
    "max_tokens": 512,
    # 🔗 Embedding Settings
    "embedding_model": "nomic-embed-text:latest",
    # ✂️ Chunking Settings
    "chunking_method": "docling",
    "chunk_size": 800,
    "chunk_overlap": 100,
    "use_hierarchical_retrieval": False,
    "proposition_extraction": False,
    "multimodal_enabled": False,
    # 🔍 Retrieval Settings
    "retrieval_k": 5,
    "search_type": "mmr",
    "similarity_threshold": 0.3,
    # ⚙️ Advanced Retrieval Features
    "use_reranking": False,
    "use_query_expansion": False,
    "use_hybrid_search": False,
    "enable_pii_redaction": False,
    # 📄 Document Processing
    "ocr_enabled": False,
    "extract_tables": False,
    # 💾 Storage
    "storage_path": None,
    # 🏗️ RAG ARCHITECTURES
    "enable_memory": True,
    "enable_adaptive": False,
    "enable_corrective": False,
    "enable_self_rag": False,
    "enable_hyde": False,
    "enable_branched": False,
    "enable_agentic": False,
    # 🧪 Evaluation
    "enable_evaluation": False,
    # 🚀 ADVANCED FEATURES
    "enable_query_caching": False,
    "enable_query_decomposition": False,
    # 📝 PROMPT ENGINEERING
    "use_dynamic_prompting": False,
}

# ===== NEW: EVALUATION CONFIGURATION =====
EVALUATION_CONFIG = {
    # Core toggles
    "enable_evaluation": True,
    "enable_real_time_metrics": True,
    "enable_quality_scoring": True,
    "enable_cost_tracking": True,
    "enable_failure_analysis": True,
    "enable_user_feedback": True,
    
    # Evaluation modes
    "evaluation_mode": "production",  # "production", "testing", "benchmark"
    
    # Quality thresholds
    "min_faithfulness_score": 0.7,
    "min_relevance_score": 0.7,
    "max_hallucination_tolerance": 0.1,
    "min_context_precision": 0.6,
    
    # Performance thresholds
    "max_latency_ms": 5000,
    "max_tokens_per_query": 4000,
    "target_cache_hit_rate": 0.3,
    "p95_latency_target_ms": 3000,
    
    # Storage paths
    "log_all_queries": True,
    "log_file_path": "logs/evaluation_logs.jsonl",
    "metrics_db_path": "data/metrics.db",
    "golden_dataset_path": "data/golden_dataset.json",
    
    # Golden dataset settings
    "auto_generate_test_cases": False,
    "golden_dataset_size": 50,
    
    # Evaluation frequency
    "batch_eval_frequency": "daily",  # "hourly", "daily", "weekly"
    "regression_test_on_startup": False,
    "eval_sample_rate": 1.0,  # 1.0 = evaluate all queries, 0.1 = 10% sampling
    
    # LLM for evaluation (can be cheaper/faster model)
    "eval_llm_model": "gemma2:2b",  # Use same model or specify different
    "eval_llm_temperature": 0.0,  # Lower temp for consistent evaluation
    
    # Dashboard settings
    "show_eval_dashboard": True,
    "dashboard_refresh_interval": 5,  # seconds
    "metrics_retention_days": 30,  # How long to keep metrics
    
    # Component tracking
    "track_component_latency": True,
    "track_token_usage": True,
    "track_cache_performance": True,
    
    # A/B Testing
    "enable_ab_testing": False,
    "ab_test_split": 0.5,  # 50/50 split
    
    # Alerting thresholds
    "alert_on_high_latency": True,
    "alert_on_low_quality": True,
    "alert_on_high_error_rate": True,
    "error_rate_threshold": 0.05,  # 5% error rate triggers alert
}

# Evaluation prompt templates
EVAL_PROMPTS = {
    "faithfulness": """Given the following CONTEXT and ANSWER, determine if the answer is faithful to the context.
An answer is faithful if all claims in the answer can be directly supported by the context.
Do not consider external knowledge. Only check if the answer is supported by the given context.

CONTEXT:
{context}

ANSWER:
{answer}

Respond ONLY with a JSON object in this exact format:
{{"score": 0.85, "reasoning": "The answer is mostly supported by the context..."}}

Score should be between 0.0 (completely unfaithful) and 1.0 (completely faithful).""",

    "relevance": """Given the following QUESTION and ANSWER, determine if the answer is relevant to the question.
An answer is relevant if it directly addresses what was asked, even if it says "I don't know".

QUESTION:
{question}

ANSWER:
{answer}

Respond ONLY with a JSON object in this exact format:
{{"score": 0.90, "reasoning": "The answer directly addresses the question..."}}

Score should be between 0.0 (completely irrelevant) and 1.0 (perfectly relevant).""",

    "hallucination": """Given the following CONTEXT and ANSWER, identify any hallucinations.
A hallucination is a claim in the answer that is NOT supported by the context.
List specific claims that appear in the answer but are not found in the context.

CONTEXT:
{context}

ANSWER:
{answer}

Respond ONLY with a JSON object in this exact format:
{{"hallucinations": ["claim1", "claim2"], "has_hallucination": true, "severity": "high"}}

If no hallucinations found, return: {{"hallucinations": [], "has_hallucination": false, "severity": "none"}}
Severity can be: "none", "low", "medium", "high".""",

    "context_relevance": """Given the following QUESTION and CONTEXT CHUNK, determine if this chunk is relevant to answering the question.

QUESTION:
{question}

CONTEXT CHUNK:
{context_chunk}

Respond ONLY with a JSON object in this exact format:
{{"is_relevant": true, "score": 0.75, "reasoning": "This chunk contains information about..."}}

Score should be between 0.0 (completely irrelevant) and 1.0 (highly relevant).""",

    "answer_completeness": """Given the following QUESTION, ANSWER, and CONTEXT, determine if the answer is complete.
A complete answer addresses all parts of the question using the available context.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Respond ONLY with a JSON object in this exact format:
{{"score": 0.80, "missing_aspects": ["aspect1", "aspect2"], "reasoning": "The answer covers most points but..."}}

Score should be between 0.0 (incomplete) and 1.0 (fully complete).""",
}

# Metric definitions for dashboard
METRIC_DEFINITIONS = {
    "faithfulness": {
        "name": "Faithfulness",
        "description": "How well the answer is grounded in retrieved context",
        "range": [0.0, 1.0],
        "good_threshold": 0.7,
        "format": "percentage"
    },
    "relevance": {
        "name": "Answer Relevance",
        "description": "How well the answer addresses the question",
        "range": [0.0, 1.0],
        "good_threshold": 0.7,
        "format": "percentage"
    },
    "context_precision": {
        "name": "Context Precision",
        "description": "Percentage of retrieved chunks that are relevant",
        "range": [0.0, 1.0],
        "good_threshold": 0.6,
        "format": "percentage"
    },
    "latency_ms": {
        "name": "Response Latency",
        "description": "Time taken to generate response",
        "range": [0, 10000],
        "good_threshold": 3000,
        "format": "milliseconds"
    },
    "tokens_used": {
        "name": "Tokens Used",
        "description": "Total tokens consumed per query",
        "range": [0, 5000],
        "good_threshold": 2000,
        "format": "count"
    },
    "cache_hit_rate": {
        "name": "Cache Hit Rate",
        "description": "Percentage of queries served from cache",
        "range": [0.0, 1.0],
        "good_threshold": 0.3,
        "format": "percentage"
    },
}

# Failure categories for analysis
FAILURE_CATEGORIES = {
    "retrieval_failure": "Relevant documents not retrieved",
    "comprehension_failure": "Context misunderstood by LLM",
    "generation_failure": "Poor quality answer generated",
    "relevance_failure": "Answer doesn't address the question",
    "hallucination": "Answer contains unsupported claims",
    "timeout": "Query exceeded time limit",
    "error": "System error occurred",
}

# Component names for tracking
COMPONENT_NAMES = {
    "retrieval": "Document Retrieval",
    "generation": "Answer Generation",
    "evaluation": "Quality Evaluation",
    "reranking": "Document Reranking",
    "query_expansion": "Query Expansion",
    "adaptive_routing": "Adaptive Routing",
    "corrective_rag": "Corrective RAG",
    "self_rag": "Self-RAG Iteration",
    "hyde": "HyDE Generation",
    "branched_rag": "Branched Routing",
    "agentic_rag": "Agentic Synthesis",
}