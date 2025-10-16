import time
from typing import Dict, List, Any, Optional
from langchain_ollama import ChatOllama
from config import EVALUATION_CONFIG, EVAL_PROMPTS, FAILURE_CATEGORIES, COMPONENT_NAMES
from utils import (
    generate_query_id,
    safe_json_parse,
    extract_score_from_text,
    classify_failure,
    aggregate_metrics,
    create_evaluation_summary,
    log_to_jsonl,
    load_jsonl,
    format_timestamp,
    calculate_token_estimate,
    logger
)

class RAGEvaluator:
    """Evaluates RAG system responses for quality, performance, and failures"""
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or EVALUATION_CONFIG
        self.eval_llm = ChatOllama(model=self.config.get("eval_llm_model", "llama3.1"))
        self.session_metrics = []
        self.cache_hits = 0
        self.total_queries = 0
        self.component_latencies = {name: [] for name in COMPONENT_NAMES}
        self.log_path = self.config.get("log_path", "logs/evaluation_logs.jsonl")

    def evaluate_response(
        self,
        query: str,
        answer: str,
        retrieved_contexts: List[str],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Evaluate a single response for quality and performance"""
        start_time = time.time()
        query_id = generate_query_id(query)
        results = {"query_id": query_id, "timestamp": format_timestamp()}
        metadata = metadata or {}

        try:
            # Faithfulness
            faith_prompt = EVAL_PROMPTS["faithfulness"].format(
                context="\n".join(retrieved_contexts[:3]),
                answer=answer
            )
            faith_resp = self.eval_llm.invoke(faith_prompt).content.strip()
            results["faithfulness"] = extract_score_from_text(faith_resp)

            # Relevance
            relev_prompt = EVAL_PROMPTS["relevance"].format(
                question=query,
                answer=answer
            )
            relev_resp = self.eval_llm.invoke(relev_prompt).content.strip()
            results["relevance"] = extract_score_from_text(relev_resp)

            # Hallucination
            hall_prompt = EVAL_PROMPTS["hallucination"].format(
                context="\n".join(retrieved_contexts[:3]),
                answer=answer
            )
            hall_resp = self.eval_llm.invoke(hall_prompt).content.strip()
            results["has_hallucination"] = "yes" in hall_resp.lower()
            results["hallucination_severity"] = safe_json_parse(hall_resp).get("severity", "none")

            # Context Precision
            context_relevance = []
            for ctx in retrieved_contexts:
                if not ctx.strip():
                    context_relevance.append(0.0)
                    continue
                ctx_prompt = EVAL_PROMPTS["context_relevance"].format(
                    question=query,
                    context=ctx[:500]
                )
                ctx_resp = self.eval_llm.invoke(ctx_prompt).content.strip()
                context_relevance.append(extract_score_from_text(ctx_resp))
            results["context_precision"] = sum(context_relevance) / len(context_relevance) if context_relevance else 0.0

            # Performance Metrics
            results["eval_latency_ms"] = (time.time() - start_time) * 1000
            results["total_latency_ms"] = metadata.get("latency_ms", 0)

            # Failure Analysis
            results["failure_category"] = classify_failure(
                faithfulness=results["faithfulness"],
                relevance=results["relevance"],
                has_hallucination=results["has_hallucination"],
                latency_ms=results["total_latency_ms"],
                error=None,
                thresholds=self.config
            )

            # Additional Metadata
            results.update({
                "retrieval_method_used": metadata.get("retrieval_method_used", "unknown"),
                "config_features": metadata.get("config_features", []),
                "token_estimate": calculate_token_estimate(query + answer + "\n".join(retrieved_contexts))
            })

            # Log if enabled
            if self.config.get("log_all_queries", False):
                log_to_jsonl(results, self.log_path)

            # Update session metrics
            self.session_metrics.append(results)
            self.total_queries += 1

            return results

        except Exception as e:
            logger.error(f"Evaluation failed for query {query_id}: {e}")
            results.update({
                "faithfulness": 0.0,
                "relevance": 0.0,
                "context_precision": 0.0,
                "has_hallucination": False,
                "hallucination_severity": "none",
                "failure_category": "error",
                "error_message": str(e),
                "eval_latency_ms": (time.time() - start_time) * 1000
            })
            log_to_jsonl(results, self.log_path)
            return results

    def run_batch_evaluation(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run evaluation on a batch of test cases"""
        batch_results = []
        for test_case in test_cases:
            query = test_case.get("query", "")
            answer = test_case.get("answer", "")
            contexts = test_case.get("contexts", [])
            metadata = test_case.get("metadata", {})
            result = self.evaluate_response(query, answer or "N/A", contexts, metadata)
            batch_results.append(result)

        aggregated = aggregate_metrics(batch_results)
        summary = create_evaluation_summary(aggregated, self.config)
        summary["individual_results"] = batch_results
        return summary

    def track_component_latency(self, component: str, latency_ms: float):
        """Track latency for specific components"""
        if component in self.component_latencies:
            self.component_latencies[component].append(latency_ms)

    def track_cache_hit(self, cache_hit: bool):
        """Track cache hit/miss"""
        if cache_hit:
            self.cache_hits += 1

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session-level statistics"""
        return {
            "total_queries": self.total_queries,
            "cache_hit_rate": self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0,
            "failure_rate": len([m for m in self.session_metrics if m.get("failure_category")]) / self.total_queries if self.total_queries > 0 else 0.0,
            "avg_metrics": aggregate_metrics(self.session_metrics)
        }

    def get_real_time_metrics(self, window_minutes: int = 10) -> Dict[str, Any]:
        """Get metrics for recent queries within a time window"""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [
            m for m in self.session_metrics
            if datetime.strptime(m["timestamp"], '%Y-%m-%d %H:%M:%S').timestamp() > cutoff_time
        ]
        return aggregate_metrics(recent_metrics)

    def check_alerts(self) -> List[str]:
        """Check for quality or performance issues and return alerts"""
        alerts = []
        recent_metrics = self.get_real_time_metrics(window_minutes=10)
        if self.config.get("alert_on_low_quality", False):
            if recent_metrics.get("avg_faithfulness", 1.0) < self.config.get("min_faithfulness_score", 0.7):
                alerts.append("Low faithfulness detected")
            if recent_metrics.get("avg_relevance", 1.0) < self.config.get("min_relevance_score", 0.7):
                alerts.append("Low relevance detected")
        if self.config.get("alert_on_high_error_rate", False):
            if recent_metrics.get("failure_rate", 0.0) > self.config.get("error_rate_threshold", 0.05):
                alerts.append("High error rate detected")
        return alerts