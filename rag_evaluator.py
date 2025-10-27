"""
Advanced RAG Evaluator with comprehensive metrics
"""

import time
import json
from typing import Dict, List, Optional, Any
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from utils import logger

class RAGEvaluator:
    """Evaluates RAG pipeline performance with advanced metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = ChatOllama(
            model=config.get("eval_llm_model", "gemma2:2b"),
            temperature=0.0,
            num_predict=500
        )
        self.embeddings = OllamaEmbeddings(model=config.get("embedding_model", "nomic-embed-text"))
        self.total_queries = 0
        self.cache_hits = 0
        self.failures = 0
        self.latency_metrics = {
            "retrieval": [],
            "generation": [],
            "evaluation": []
        }
    
    def evaluate_response(self, query: str, answer: str, retrieved_contexts: List[str], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate a single RAG response"""
        start_time = time.time()
        results = {"timestamp": start_time}
        metadata = metadata or {}
        
        try:
            faith_prompt = PromptTemplate.from_template(
                """Is the ANSWER fully supported by the CONTEXT? Rate 0-1 and explain.
CONTEXT:
{context}
ANSWER:
{answer}
Respond in JSON:
{{
  "score": float,
  "explanation": str
}}"""
            )
            
            faith_context = " ".join(retrieved_contexts[:3])[:1000]
            faith_response = self.llm.invoke(faith_prompt.format(context=faith_context, answer=answer))
            faith_data = self._parse_json_response(faith_response.content)
            results["faithfulness"] = faith_data.get("score", 0.0)
            results["faithfulness_explanation"] = faith_data.get("explanation", "N/A")
            
            relev_prompt = PromptTemplate.from_template(
                """Is the ANSWER relevant to the QUESTION? Rate 0-1 and explain.
QUESTION:
{question}
ANSWER:
{answer}
Respond in JSON:
{{
  "score": float,
  "explanation": str
}}"""
            )
            
            relev_response = self.llm.invoke(relev_prompt.format(question=query, answer=answer))
            relev_data = self._parse_json_response(relev_response.content)
            results["relevance"] = relev_data.get("score", 0.0)
            results["relevance_explanation"] = relev_data.get("explanation", "N/A")
            
            context_relevance = []
            for ctx in retrieved_contexts:
                if not ctx.strip():
                    context_relevance.append(0.0)
                    continue
                
                ctx_prompt = PromptTemplate.from_template(
                    """Is this CONTEXT relevant to the QUESTION? Rate 0-1.
QUESTION:
{question}
CONTEXT:
{context}
Respond with a number between 0 and 1"""
                )
                
                try:
                    ctx_response = self.llm.invoke(ctx_prompt.format(question=query, context=ctx[:500]))
                    score = float(ctx_response.content.strip())
                    context_relevance.append(score)
                except:
                    context_relevance.append(0.0)
            
            results["context_precision"] = sum(context_relevance) / len(context_relevance) if context_relevance else 0.0
            
            halluc_prompt = PromptTemplate.from_template(
                """Does the ANSWER contain claims not supported by the CONTEXT? If yes, identify them.
CONTEXT:
{context}
ANSWER:
{answer}
Respond in JSON:
{{
  "has_hallucination": bool,
  "hallucinated_claims": str,
  "severity": str
}}"""
            )
            
            halluc_response = self.llm.invoke(halluc_prompt.format(context=faith_context, answer=answer))
            halluc_data = self._parse_json_response(halluc_response.content)
            results["has_hallucination"] = halluc_data.get("has_hallucination", False)
            results["hallucinated_claims"] = halluc_data.get("hallucinated_claims", "None")
            results["hallucination_severity"] = halluc_data.get("severity", "none")
            
            try:
                if len(retrieved_contexts) > 1:
                    ctx_embs = self.embeddings.embed_documents([c for c in retrieved_contexts if c.strip()])
                    if len(ctx_embs) > 1:
                        sims = []
                        for i in range(len(ctx_embs)):
                            for j in range(i + 1, len(ctx_embs)):
                                sim = cosine_similarity([ctx_embs[i]], [ctx_embs[j]])[0][0]
                                sims.append(sim)
                        results["context_diversity"] = 1 - (sum(sims) / len(sims)) if sims else 1.0
                    else:
                        results["context_diversity"] = 1.0
                else:
                    results["context_diversity"] = 1.0
            except Exception as e:
                logger.warning(f"Context diversity eval failed: {e}")
                results["context_diversity"] = "N/A"
            
            results["metadata"] = metadata
            results["failure_category"] = self._detect_failure_category(results)
            results["eval_latency_ms"] = (time.time() - start_time) * 1000
            
            self.total_queries += 1
            if results["faithfulness"] < 0.5 or results["relevance"] < 0.5 or results["has_hallucination"]:
                self.failures += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self.failures += 1
            results["error"] = str(e)
            results["eval_latency_ms"] = (time.time() - start_time) * 1000
            return results
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response from LLM, with fallback"""
        try:
            return json.loads(response)
        except:
            logger.warning(f"Failed to parse JSON: {response}")
            return {}
    
    def _detect_failure_category(self, results: Dict) -> str:
        """Determine failure category based on metrics"""
        if results.get("error"):
            return "system_error"
        if results.get("has_hallucination", False):
            return "hallucination"
        if results.get("faithfulness", 1.0) < 0.5:
            return "low_faithfulness"
        if results.get("relevance", 1.0) < 0.5:
            return "low_relevance"
        if results.get("context_precision", 1.0) < 0.5:
            return "low_context_precision"
        return "none"
    
    def track_component_latency(self, component: str, latency_ms: float):
        """Track latency for retrieval, generation, or evaluation"""
        if component in self.latency_metrics:
            self.latency_metrics[component].append(latency_ms)
    
    def track_cache_hit(self, cache_hit: bool):
        """Track cache hit rate"""
        if cache_hit:
            self.cache_hits += 1
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Return session-wide statistics"""
        return {
            "total_queries": self.total_queries,
            "cache_hit_rate": self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0,
            "failure_rate": self.failures / self.total_queries if self.total_queries > 0 else 0.0,
            "avg_retrieval_latency_ms": sum(self.latency_metrics["retrieval"]) / len(self.latency_metrics["retrieval"]) if self.latency_metrics["retrieval"] else 0.0,
            "avg_generation_latency_ms": sum(self.latency_metrics["generation"]) / len(self.latency_metrics["generation"]) if self.latency_metrics["generation"] else 0.0,
            "avg_evaluation_latency_ms": sum(self.latency_metrics["evaluation"]) / len(self.latency_metrics["evaluation"]) if self.latency_metrics["evaluation"] else 0.0
        }
    
    def check_alerts(self) -> List[str]:
        """Check for performance issues and return alerts"""
        alerts = []
        stats = self.get_session_stats()
        
        if stats["failure_rate"] > 0.3:
            alerts.append(f"High failure rate: {stats['failure_rate']:.1%}")
        
        if stats["cache_hit_rate"] < 0.1 and self.total_queries > 10:
            alerts.append("Low cache hit rate: consider enabling caching")
        
        if stats["avg_evaluation_latency_ms"] > 5000:
            alerts.append("High evaluation latency: consider disabling advanced evaluation")
        
        return alerts
    
    def run_batch_evaluation(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run evaluation on a batch of test cases"""
        results = []
        for case in test_cases:
            result = self.evaluate_response(
                query=case["query"],
                answer=case.get("answer", ""),
                retrieved_contexts=case["contexts"],
                metadata=case.get("metadata", {})
            )
            results.append(result)
        
        return {
            "batch_results": results,
            "summary": {
                "avg_faithfulness": sum(r.get("faithfulness", 0) for r in results) / len(results) if results else 0,
                "avg_relevance": sum(r.get("relevance", 0) for r in results) / len(results) if results else 0,
                "avg_context_precision": sum(r.get("context_precision", 0) for r in results) / len(results) if results else 0,
                "hallucination_rate": sum(1 for r in results if r.get("has_hallucination", False)) / len(results) if results else 0
            }
        }
