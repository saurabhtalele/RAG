import logging
import json
import hashlib
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger("RAGApp")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            sanitized[k] = v
        else:
            sanitized[k] = str(v)
    return sanitized

def check_ollama_connection(embedding_model: str) -> bool:
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        emb = OllamaEmbeddings(model=embedding_model)
        emb.embed_query("test")
        return True
    except Exception:
        return False

# PII Redaction
_analyzer = None
_anonymizer = None

def get_pii_engines():
    global _analyzer, _anonymizer
    if _analyzer is None:
        _analyzer = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer

def redact_pii(text: str) -> str:
    if not text.strip():
        return text
    analyzer, anonymizer = get_pii_engines()
    results = analyzer.analyze(text=text, language='en')
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text

def generate_query_id(query: str) -> str:
    """Generate a unique ID for a query using its content and timestamp"""
    timestamp = str(time.time())
    query_hash = hashlib.md5((query + timestamp).encode()).hexdigest()
    return query_hash[:8]

def safe_json_parse(text: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
    """Safely parse JSON, returning default if parsing fails"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse JSON: {text}")
        return default if default is not None else {}

def extract_score_from_text(text: str) -> float:
    """Extract a numeric score from text, defaulting to 0.5"""
    import re
    match = re.search(r'\d*\.?\d+', text)
    return float(match.group()) if match else 0.5

def classify_failure(
    faithfulness: float,
    relevance: float,
    has_hallucination: bool,
    latency_ms: float,
    error: Optional[Any],
    thresholds: Dict[str, float]
) -> Optional[str]:
    """Classify failure based on metrics and thresholds"""
    if error:
        return "error"
    if latency_ms > thresholds.get("max_latency_ms", 5000):
        return "timeout"
    if has_hallucination:
        return "hallucination"
    if faithfulness < thresholds.get("min_faithfulness_score", 0.7):
        return "comprehension_failure"
    if relevance < thresholds.get("min_relevance_score", 0.7):
        return "relevance_failure"
    return None

def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics from multiple evaluations"""
    if not metrics:
        return {}
    
    aggregated = {
        "avg_faithfulness": 0.0,
        "avg_relevance": 0.0,
        "avg_context_precision": 0.0,
        "avg_latency_ms": 0.0,
        "failure_rate": 0.0,
        "hallucination_rate": 0.0
    }
    total = len(metrics)
    valid_faithfulness = [m["faithfulness"] for m in metrics if m.get("faithfulness") is not None]
    valid_relevance = [m["relevance"] for m in metrics if m.get("relevance") is not None]
    valid_context_precision = [m["context_precision"] for m in metrics if m.get("context_precision") is not None]
    valid_latency = [m["eval_latency_ms"] for m in metrics if m.get("eval_latency_ms") is not None]
    failures = [m for m in metrics if m.get("failure_category")]
    hallucinations = [m for m in metrics if m.get("has_hallucination", False)]

    if valid_faithfulness:
        aggregated["avg_faithfulness"] = sum(valid_faithfulness) / len(valid_faithfulness)
    if valid_relevance:
        aggregated["avg_relevance"] = sum(valid_relevance) / len(valid_relevance)
    if valid_context_precision:
        aggregated["avg_context_precision"] = sum(valid_context_precision) / len(valid_context_precision)
    if valid_latency:
        aggregated["avg_latency_ms"] = sum(valid_latency) / len(valid_latency)
    aggregated["failure_rate"] = len(failures) / total if total > 0 else 0.0
    aggregated["hallucination_rate"] = len(hallucinations) / total if total > 0 else 0.0

    return aggregated

def create_evaluation_summary(aggregated: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of evaluation results"""
    status = "PASS" if (
        aggregated.get("avg_faithfulness", 1.0) >= config.get("min_faithfulness_score", 0.7) and
        aggregated.get("avg_relevance", 1.0) >= config.get("min_relevance_score", 0.7) and
        aggregated.get("failure_rate", 0.0) <= config.get("error_rate_threshold", 0.05)
    ) else "FAIL"
    
    return {
        "overall_status": status,
        "summary_metrics": aggregated,
        "timestamp": format_timestamp()
    }

def log_to_jsonl(data: Dict[str, Any], file_path: str):
    """Log data to a JSONL file"""
    try:
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        logger.error(f"Failed to log to JSONL: {e}")

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file"""
    data = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSONL line: {line.strip()}")
    except FileNotFoundError:
        logger.warning(f"Log file not found: {file_path}")
    return data

def format_timestamp() -> str:
    """Format current timestamp as string"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def calculate_token_estimate(text: str) -> int:
    """Estimate token count based on word count"""
    return len(text.split()) * 2  # Rough estimate: 1 word ≈ 2 tokens