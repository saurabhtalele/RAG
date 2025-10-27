"""
Optimized RAG Engine with 8 Advanced Architectures + New Features
ALL FIXED FOR OLLAMA COMPATIBILITY WITH gemma2:2b
"""

import uuid
import tempfile
import time
import json
import hashlib
import re
from typing import Tuple, Any, Dict, List, Optional, Callable
from pathlib import Path
from collections import deque, OrderedDict
from datetime import datetime, timedelta
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from config import DEFAULT_CONFIG, EVALUATION_CONFIG
from utils import sanitize_metadata, logger
from rag_evaluator import RAGEvaluator

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not installed")

_query_cache = OrderedDict()
_cache_max_size = 100

def hash_config_for_cache(config: Dict[str, Any]) -> str:
    """Create hash from config for cache key"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def get_cached_result(query: str, config_hash: str) -> Optional[Tuple[str, List]]:
    """Retrieve cached result"""
    cache_key = f"{query}_{config_hash}"
    return _query_cache.get(cache_key)

def cache_result(query: str, config_hash: str, answer: str, sources: List):
    """Cache query result"""
    global _query_cache
    cache_key = f"{query}_{config_hash}"
    _query_cache[cache_key] = (answer, sources)
    
    if len(_query_cache) > _cache_max_size:
        _query_cache.popitem(last=False)

def log_interaction(data: Dict[str, Any]):
    """Log interaction to file"""
    try:
        with open("rag_interactions.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")

def save_feedback(feedback_data: Dict[str, Any]):
    """Save user feedback"""
    try:
        with open("rag_feedback.jsonl", "a") as f:
            f.write(json.dumps(feedback_data) + "\n")
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")

class ConversationMemoryManager:
    """Manages conversation history with token-aware context window"""
    
    def __init__(self, max_memory_tokens: int = 2000):
        self.max_memory_tokens = max_memory_tokens
        self.conversation_history = deque(maxlen=10)
        self.token_count = 0
    
    def add_turn(self, question: str, answer: str, token_counter):
        """Add Q&A pair to memory with token management"""
        tokens = token_counter(f"{question} {answer}")
        
        if self.token_count + tokens > self.max_memory_tokens:
            if self.conversation_history:
                old_q, old_a, old_t = self.conversation_history.popleft()
                self.token_count -= old_t
        
        self.conversation_history.append((question, answer, tokens))
        self.token_count += tokens
    
    def get_context_for_prompt(self) -> str:
        """Returns conversation history formatted for LLM context"""
        if not self.conversation_history:
            return ""
        
        context = "## Previous Conversation:\n"
        for q, a, _ in list(self.conversation_history)[-5:]:
            context += f"Q: {q}\nA: {a[:200]}\n"
        return context
    
    def clear(self):
        """Reset memory"""
        self.conversation_history.clear()
        self.token_count = 0

class MetadataAwareRetriever(Runnable):
    """Retriever with metadata filtering support - inherits from Runnable"""
    
    def __init__(self, base_retriever, config):
        super().__init__()
        self.base_retriever = base_retriever
        self.config = config
    
    def parse_metadata_filters(self, query: str) -> Tuple[str, Dict]:
        """Extract metadata filters from query"""
        filters = {}
        clean_query = query
        
        if match := re.search(r"page:(\d+)-(\d+)", query):
            filters["page"] = {"$gte": int(match.group(1)), "$lte": int(match.group(2))}
            clean_query = re.sub(r"page:\d+-\d+", "", clean_query).strip()
        elif match := re.search(r"page:(\d+)", query):
            filters["page"] = int(match.group(1))
            clean_query = re.sub(r"page:\d+", "", clean_query).strip()
        
        if match := re.search(r"file:([^\s]+)", query):
            filters["filename"] = {"$eq": match.group(1)}
            clean_query = re.sub(r"file:[^\s]+", "", clean_query).strip()
        
        return clean_query, filters
    
    def invoke(self, query: str, config=None, **kwargs):
        """Retrieve with metadata filtering"""
        clean_query, filters = self.parse_metadata_filters(query)
        
        if filters:
            logger.info(f"Applying metadata filters: {filters}")
            try:
                return self.base_retriever.invoke(clean_query, filter=filters)
            except:
                logger.warning("Metadata filtering failed, using base retrieval")
                return self.base_retriever.invoke(clean_query)
        
        return self.base_retriever.invoke(query)

class RRFRetriever:
    """Reciprocal Rank Fusion for combining multiple retrievers"""
    
    def __init__(self, retrievers: Dict[str, Any], k: int = 60):
        self.retrievers = retrievers
        self.k = k
    
    def invoke(self, query: str, top_k: int = 5) -> List[Document]:
        """Fuse results from multiple retrievers using RRF"""
        doc_scores = {}
        doc_objects = {}
        
        for retriever_name, retriever in self.retrievers.items():
            try:
                docs = retriever.invoke(query) if hasattr(retriever, 'invoke') else retriever(query)
                for rank, doc in enumerate(docs):
                    doc_id = doc.metadata.get("source", "") + str(hash(doc.page_content[:100]))
                    
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0
                        doc_objects[doc_id] = doc
                    
                    doc_scores[doc_id] += 1 / (rank + 1 + self.k)
            except Exception as e:
                logger.warning(f"Retriever {retriever_name} failed: {e}")
                continue
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_objects[doc_id] for doc_id, _ in sorted_docs[:top_k]]

class AdaptiveRetriever(Runnable):
    """Routes queries to different retrieval strategies based on complexity"""
    
    def __init__(self, base_retriever, llm, config):
        super().__init__()
        self.base_retriever = base_retriever
        self.llm = llm
        self.config = config
    
    def classify_query_complexity(self, query: str) -> str:
        """Classify query as simple or complex"""
        word_count = len(query.split())
        has_comparison = any(w in query.lower() for w in ["vs", "compare", "difference", "contrast"])
        has_multiple_parts = "and" in query.lower() and query.count(",") > 0
        question_marks = query.count("?")
        
        complexity_score = word_count / 10
        complexity_score += 2 if has_comparison else 0
        complexity_score += 2 if has_multiple_parts else 0
        complexity_score += 1 if question_marks > 1 else 0
        
        return "complex" if complexity_score > 3 else "simple"
    
    def retrieve_adaptive(self, query: str, k: int = 5):
        """Dynamically adjust retrieval based on complexity"""
        complexity = self.classify_query_complexity(query)
        
        if complexity == "simple":
            return self.base_retriever.invoke(query)
        else:
            initial_docs = self.base_retriever.invoke(query)
            try:
                expanded_query = expand_query(query, self.llm)
                expanded_docs = self.base_retriever.invoke(expanded_query)
                all_docs = initial_docs + expanded_docs
                unique_docs = {doc.metadata.get("source"): doc for doc in all_docs}
                return list(unique_docs.values())[:k]
            except:
                return initial_docs
    
    def invoke(self, query: str, config=None, **kwargs):
        return self.retrieve_adaptive(query, self.config.get("retrieval_k", 5))

class CorrectiveRAG:
    """Validates retrieved documents and triggers fallback if quality is low"""
    
    def __init__(self, base_retriever, llm, config):
        self.base_retriever = base_retriever
        self.llm = llm
        self.config = config
        self.relevance_threshold = 0.6
    
    def grade_retrieval_relevance(self, question: str, context: str) -> tuple:
        """Score how relevant a chunk is to the question"""
        prompt = f"""Given the QUESTION and CONTEXT, rate the relevance 0-1.
QUESTION: {question}
CONTEXT: {context[:500]}
Respond ONLY with a number between 0 and 1 (e.g., "0.85")"""
        
        try:
            response = self.llm.invoke(prompt)
            score_text = response.content.strip()
            score_match = re.search(r'0?\.\d+|[01]\.?\d*', score_text)
            if score_match:
                score = float(score_match.group())
            else:
                score = 0.5
            is_relevant = score >= self.relevance_threshold
            return is_relevant, score
        except Exception as e:
            logger.warning(f"Grading failed: {e}")
            return False, 0.0
    
    def correct_retrieval(self, question: str, initial_docs: list) -> list:
        """Grade all docs, filter poor ones, fallback if needed"""
        graded_docs = []
        for doc in initial_docs:
            is_relevant, score = self.grade_retrieval_relevance(question, doc.page_content)
            graded_docs.append({"doc": doc, "score": score, "is_relevant": is_relevant})
        
        relevant_count = len([d for d in graded_docs if d["is_relevant"]])
        coverage = relevant_count / len(graded_docs) if graded_docs else 0
        
        if coverage < 0.5:
            logger.warning(f"Low retrieval quality ({coverage:.1%}). Using fallback.")
            expanded = expand_query(question, self.llm)
            return self.base_retriever.invoke(expanded)
        
        return [d["doc"] for d in graded_docs if d["is_relevant"]]

class SelfRAG:
    """Model generates, critiques, and retrieves more if needed"""
    
    def __init__(self, base_retriever, llm, config):
        self.base_retriever = base_retriever
        self.llm = llm
        self.config = config
        self.max_iterations = 3
    
    def generate_with_self_retrieval(self, question: str, initial_context: str) -> dict:
        """Generate answer, critique it, retrieve more if needed"""
        iteration_data = []
        current_context = initial_context
        
        for iteration in range(self.max_iterations):
            answer_prompt = f"""Using this context, answer the question concisely.
CONTEXT: {current_context}
QUESTION: {question}
ANSWER:"""
            
            answer = self.llm.invoke(answer_prompt)
            answer_text = answer.content
            
            critique_prompt = f"""Evaluate this answer:
- Is it well-supported by the context?
- Should we retrieve MORE INFORMATION?

ANSWER: {answer_text}

Respond with:
SUPPORTED: yes/no
NEEDS_MORE: yes/no
RETRIEVAL_QUERY: (if yes, what to search)"""
            
            critique = self.llm.invoke(critique_prompt)
            critique_text = critique.content
            
            iteration_data.append({
                "iteration": iteration + 1,
                "answer": answer_text,
                "critique": critique_text
            })
            
            if "NEEDS_MORE: yes" in critique_text:
                lines = critique_text.split("\n")
                query_lines = [l for l in lines if l.startswith("RETRIEVAL_QUERY:")]
                if query_lines:
                    new_query = query_lines[0].replace("RETRIEVAL_QUERY:", "").strip()
                    new_docs = self.base_retriever.invoke(new_query)
                    new_context = "\n".join([doc.page_content for doc in new_docs[:2]])
                    current_context += f"\n[Additional context from iteration {iteration+1}]:\n{new_context}"
                    continue
            break
        
        return {
            "final_answer": iteration_data[-1]["answer"],
            "total_iterations": len(iteration_data),
            "iterations": iteration_data
        }

class HyDERetriever:
    """Generates hypothetical docs to improve retrieval"""
    
    def __init__(self, base_retriever, llm, config):
        self.base_retriever = base_retriever
        self.llm = llm
        self.config = config
    
    def generate_hypothetical_answer(self, question: str) -> str:
        """Generate hypothetical answer to guide retrieval"""
        prompt = f"""Generate a concise hypothetical answer to this question (max 100 words):
QUESTION: {question}
HYPOTHETICAL ANSWER:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"HyDe generation failed: {e}")
            return question
    
    def retrieve_with_hyde(self, question: str) -> Tuple[List[Document], str]:
        """Retrieve using hypothetical answer"""
        hypothetical = self.generate_hypothetical_answer(question)
        docs = self.base_retriever.invoke(hypothetical)
        return docs, hypothetical

class BranchedRAG:
    """Routes queries to specialized retrievers based on domain"""
    
    def __init__(self, base_retriever, llm, config):
        self.base_retriever = base_retriever
        self.llm = llm
        self.config = config
    
    def classify_query_domain(self, query: str) -> str:
        """Classify query domain"""
        prompt = f"""Classify this query into a domain (technical, legal, general):
QUERY: {query}
Respond with one word: technical/legal/general"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip().lower()
        except:
            return "general"
    
    def retrieve_branched(self, query: str) -> Tuple[List[Document], str]:
        """Retrieve with domain-specific strategy"""
        domain = self.classify_query_domain(query)
        
        if domain == "technical":
            k = self.config.get("retrieval_k", 5) + 2
        elif domain == "legal":
            k = self.config.get("retrieval_k", 5) + 3
        else:
            k = self.config.get("retrieval_k", 5)
        
        docs = self.base_retriever.invoke(query)[:k]
        return docs, domain

class DocumentAgent:
    """Agent that specializes in one document"""
    
    def __init__(self, document: Document, llm, config):
        self.document = document
        self.llm = llm
        self.config = config
    
    def answer_from_document(self, question: str) -> Optional[str]:
        """Answer question using only this document"""
        prompt = f"""Using ONLY this document, answer the question. If the document doesn't have the answer, say "I don't know".
DOCUMENT: {self.document.page_content[:1000]}
QUESTION: {question}
ANSWER:"""
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip()
            return answer if answer != "I don't know" else None
        except Exception as e:
            logger.warning(f"Document agent failed: {e}")
            return None

class MetaAgent:
    """Coordinates multiple document agents"""
    
    def __init__(self, base_retriever, llm, config):
        self.base_retriever = base_retriever
        self.llm = llm
        self.config = config
    
    def synthesize_answer(self, question: str) -> str:
        """Synthesize answer from multiple document agents"""
        docs = self.base_retriever.invoke(question)[:self.config.get("retrieval_k", 5)]
        agents = [DocumentAgent(doc, self.llm, self.config) for doc in docs]
        
        answers = []
        for agent in agents:
            ans = agent.answer_from_document(question)
            if ans:
                answers.append(ans)
        
        if not answers:
            return "I don't know"
        
        synthesis_prompt = f"""Synthesize a concise answer from these responses:
RESPONSES:
{chr(10).join([f'- {ans}' for ans in answers])}

QUESTION: {question}
FINAL ANSWER:"""
        
        try:
            response = self.llm.invoke(synthesis_prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")
            return answers[0] if answers else "I don't know"

def build_llm(config: Dict[str, Any]) -> ChatOllama:
    """Build LLM with config - FIXED for Ollama API"""
    model_name = config.get("llm_model", "gemma2:2b")
    temp = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 512)
    
    logger.info(f"Building LLM: model={model_name}, temp={temp}, num_predict={max_tokens}")
    
    try:
        return ChatOllama(
            model=model_name,
            temperature=temp,
            num_predict=max_tokens
        )
    except Exception as e:
        logger.error(f"Failed to build LLM: {e}")
        raise

def create_embeddings(model_name: str) -> OllamaEmbeddings:
    """Create embeddings with error handling"""
    try:
        return OllamaEmbeddings(model=model_name)
    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        raise

def build_vectorstore(texts: List[str], metadatas: List[Dict], embeddings, persist_dir: str) -> Chroma:
    """Build Chroma vectorstore"""
    try:
        return Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name="rag_collection"
        )
    except Exception as e:
        logger.error(f"Failed to build vectorstore: {e}")
        raise

def build_retriever(vectorstore: Chroma, documents: List[Document], config: Dict[str, Any], llm) -> Tuple[Any, Any]:
    """Build retriever based on config"""
    k = config.get("retrieval_k", 5)
    search_type = config.get("search_type", "similarity")
    search_kwargs = {"k": k}
    
    if search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = config.get("similarity_threshold", 0.3)
    
    base_retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    
    if config.get("use_reranking", False):
        compressor = LLMChainExtractor.from_llm(llm)
        base_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    
    if config.get("use_rrf_fusion", False):
        retrievers = {"vector": base_retriever}
        
        if config.get("use_hybrid_search", False):
            tokenized_corpus = [doc.page_content.split() for doc in documents]
            bm25 = BM25Okapi(tokenized_corpus)
            
            def bm25_retriever(query: str) -> List[Document]:
                bm25_scores = bm25.get_scores(query.split())
                doc_scores = {i: score for i, score in enumerate(bm25_scores)}
                sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                return [documents[i] for i, _ in sorted_docs[:k]]
            
            retrievers["bm25"] = RunnableLambda(bm25_retriever)
        
        rrf_retriever = RRFRetriever(retrievers)
        retriever_for_chain = RunnableLambda(lambda q: rrf_retriever.invoke(q, k))
    
    elif config.get("use_hybrid_search", False):
        tokenized_corpus = [doc.page_content.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        
        def hybrid_retriever(query: str) -> List[Document]:
            vector_docs = base_retriever.invoke(query)
            bm25_scores = bm25.get_scores(query.split())
            doc_scores = {i: score for i, score in enumerate(bm25_scores)}
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            bm25_indices = [i for i, _ in sorted_docs[:k]]
            bm25_docs = [documents[i] for i in bm25_indices]
            combined_docs = {doc.metadata.get("source"): doc for doc in vector_docs + bm25_docs}
            return list(combined_docs.values())[:k]
        
        retriever_for_chain = RunnableLambda(hybrid_retriever)
    else:
        retriever_for_chain = base_retriever
    
    if config.get("enable_metadata_filtering", False):
        retriever_for_chain = MetadataAwareRetriever(retriever_for_chain, config)
    
    logger.debug(f"Built retriever: base={type(base_retriever)}, chain={type(retriever_for_chain)}")
    return base_retriever, retriever_for_chain

def expand_query(query: str, llm) -> str:
    """Expand query for better retrieval"""
    prompt = f"""Generate an expanded version of this query to improve document retrieval.
Original Query: {query}
Expanded Query:"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return query

def build_rag_chain(retriever, llm, config: Dict[str, Any], memory_manager: Optional[ConversationMemoryManager] = None):
    """Build RAG chain with dynamic prompting"""
    chunk_types = "text, tables, images" if config.get("multimodal_enabled") else "text"
    instruction_text = "Answer in markdown format with clear sections." if config.get("use_dynamic_prompting") else ""
    
    if config.get("enable_memory", False) and memory_manager:
        def format_docs_with_memory(docs):
            memory_context = memory_manager.get_context_for_prompt()
            doc_text = "\n\n".join([doc.page_content for doc in docs])
            if memory_context:
                return f"{memory_context}\n\n---\n\n{doc_text}"
            return doc_text
        
        prompt_template = f"""You are a helpful assistant. Use the context below to answer the question accurately.
Note: Context may contain {chunk_types}.
{instruction_text}
If unsure, say "I don't know".

Context: {{context}}
Question: {{question}}
Answer:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        return (
            {"context": retriever | RunnableLambda(format_docs_with_memory), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    else:
        prompt_template = f"""You are a helpful assistant. Use the context to answer the question accurately.
Note: Context may contain {chunk_types}.
{instruction_text}
If unsure, say "I don't know".

Context: {{context}}
Question: {{question}}
Answer:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

def setup_rag_chain(
    config: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    memory_manager: Optional[ConversationMemoryManager] = None,
) -> Tuple:
    """Initialize complete RAG pipeline"""
    valid_chunks = [c for c in chunks if c.get("content", "").strip()]
    
    if not valid_chunks:
        raise ValueError("No valid text chunks found!")
    
    for c in valid_chunks:
        c["metadata"] = sanitize_metadata(c.get("metadata", {}))
    
    llm = build_llm(config)
    documents = [Document(page_content=c["content"], metadata=c["metadata"]) for c in valid_chunks]
    embeddings = create_embeddings(config["embedding_model"])
    persist_dir = config.get("storage_path") or tempfile.mkdtemp()
    
    texts = [c["content"] for c in valid_chunks]
    metadatas = [c["metadata"] for c in valid_chunks]
    
    vectorstore = build_vectorstore(texts, metadatas, embeddings, persist_dir)
    base_retriever, retriever_for_chain = build_retriever(vectorstore, documents, config, llm)
    
    graph_retriever = None
    if config.get("enable_knowledge_graph", False):
        try:
            from graph_retriever import KnowledgeGraphRetriever
            graph_retriever = KnowledgeGraphRetriever(valid_chunks)
            logger.info("Knowledge graph retriever initialized")
        except Exception as e:
            logger.warning(f"Knowledge graph initialization failed: {e}")
    
    chain = build_rag_chain(retriever_for_chain, llm, config, memory_manager)
    
    logger.debug(f"RAG chain setup complete")
    return chain, retriever_for_chain, base_retriever, vectorstore, llm, graph_retriever

def evaluate_rag_response_full(
    question: str,
    answer: str,
    contexts: List[str],
    llm: ChatOllama,
    embeddings: OllamaEmbeddings,
    start_time: float
) -> Dict[str, Any]:
    """Comprehensive RAG evaluation"""
    results = {"timestamp": time.time(), "latency_sec": round(time.time() - start_time, 2)}
    
    try:
        faith_prompt = f"""Is the ANSWER fully supported by the CONTEXT? Answer only "yes" or "no".
CONTEXT: {' '.join(contexts[:3])}
ANSWER: {answer}"""
        
        faith_resp = llm.invoke(faith_prompt).content.strip().lower()
        results["faithfulness"] = "yes" if "yes" in faith_resp else "no"
    except:
        results["faithfulness"] = "N/A"
    
    try:
        relev_prompt = f"""Is the ANSWER relevant to the QUESTION? Answer only "yes" or "no".
QUESTION: {question}
ANSWER: {answer}"""
        
        relev_resp = llm.invoke(relev_prompt).content.strip().lower()
        results["answer_relevance"] = "yes" if "yes" in relev_resp else "no"
    except:
        results["answer_relevance"] = "N/A"
    
    context_relevance = []
    for ctx in contexts:
        if not ctx.strip():
            context_relevance.append(False)
            continue
        
        try:
            ctx_prompt = f"""Is this CONTEXT relevant to the QUESTION? Answer "yes" or "no".
QUESTION: {question}
CONTEXT: {ctx[:500]}"""
            resp = llm.invoke(ctx_prompt).content.strip().lower()
            context_relevance.append("yes" in resp)
        except:
            context_relevance.append(False)
    
    results["context_relevance"] = context_relevance
    results["context_precision"] = round(sum(context_relevance) / len(context_relevance), 2) if context_relevance else 0.0
    
    return results
