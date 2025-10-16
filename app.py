import streamlit as st
import os
import tempfile
import time
import json
from pathlib import Path
from typing import List
from config import DEFAULT_CONFIG, EVALUATION_CONFIG
from utils import check_ollama_connection, logger
from pdf_processor import docling_chunks_from_pdfs
from rag_engine import (
    setup_rag_chain, expand_query, evaluate_rag_response_full,
    create_embeddings, save_feedback, log_interaction,
    ConversationMemoryManager, AdaptiveRetriever, CorrectiveRAG,
    SelfRAG, HyDERetriever, BranchedRAG, DocumentAgent, MetaAgent,
    get_cached_result, cache_result, hash_config_for_cache
)
from rag_evaluator import RAGEvaluator

# ==================== SESSION STATE INITIALIZATION ====================
for key in ["chat_history", "rag_chain", "base_retriever", "vectorstore", "llm",
            "documents_loaded", "memory_manager", "adaptive_retriever",
            "corrective_rag", "self_rag", "hyde_retriever", "branched_rag", 
            "agentic_rag", "rag_evaluator"]:
    if key not in st.session_state:
        if key == "chat_history":
            st.session_state[key] = []
        elif key == "memory_manager":
            st.session_state[key] = ConversationMemoryManager(max_memory_tokens=2000)
        else:
            st.session_state[key] = None

# Initialize cache hash in session state
if "config_hash" not in st.session_state:
    st.session_state.config_hash = hash_config_for_cache(DEFAULT_CONFIG)

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="📚 Advanced RAG Playground", page_icon="📚", layout="wide")
st.title("📚 Advanced RAG Playground")
st.caption("Local AI + Smart PDFs + 8 RAG Architectures – All on your machine!")
st.write("Debug: Main app rendering")

# ==================== SIDEBAR CONFIGURATION ====================
with st.sidebar:
    st.header("⚙️ Configuration")
    # Initialize evaluator at the start to ensure it's ready
    if "rag_evaluator" not in st.session_state and DEFAULT_CONFIG["enable_evaluation"]:
        try:
            st.session_state.rag_evaluator = RAGEvaluator(config=EVALUATION_CONFIG)
            st.info("🧪 RAG Evaluator initialized for advanced metrics.")
        except Exception as e:
            st.error(f"Failed to initialize RAG Evaluator: {e}")
            logger.error(f"Evaluator init failed: {e}")

    # Connection Status
    with st.expander("🌐 Ollama Status", expanded=True):
        if st.button("Check Connection"):
            with st.spinner("Testing..."):
                if check_ollama_connection(DEFAULT_CONFIG["embedding_model"]):
                    st.success("✅ Ollama is ready!")
                else:
                    st.error("❌ Start Ollama: `ollama serve`")

    # Storage
    use_persistent = st.checkbox("💾 Persistent Storage", value=True)
    storage_path = st.text_input(
        "Storage Path",
        value=os.path.join(os.getcwd(), "chromadb_storage") if use_persistent else "",
        disabled=not use_persistent
    ) if use_persistent else None

    # Model Settings
    llm_model = st.text_input("🤖 LLM Model", value=DEFAULT_CONFIG["llm_model"])
    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, DEFAULT_CONFIG["temperature"])
    max_tokens = st.slider("📝 Max Tokens", 128, 2048, DEFAULT_CONFIG["max_tokens"])
    embedding_model = st.text_input("📘 Embedding Model", value=DEFAULT_CONFIG["embedding_model"])

    # Chunking
    st.divider()
    st.subheader("✂️ Chunking Settings")
    chunking_method = st.selectbox("Chunking Method",
        ["docling", "recursive", "semantic", "llm_powered", "hybrid"])
    chunk_size = st.slider("📄 Chunk Size", 128, 2048, DEFAULT_CONFIG["chunk_size"])
    chunk_overlap = st.slider("🔗 Overlap", 0, 512, DEFAULT_CONFIG["chunk_overlap"])

    # Advanced Chunking
    use_hierarchical_retrieval = st.checkbox("📚 Hierarchical Retrieval")
    proposition_extraction = st.checkbox("🧠 Proposition Extraction")
    multimodal_enabled = st.checkbox("🖼️ Multimodal (Tables/Images)")
    ocr_enabled = st.checkbox("🖼️ Enable OCR (scanned PDFs)")
    extract_tables = st.checkbox("📊 Extract Tables")

    # Retrieval
    st.divider()
    st.subheader("🔍 Retrieval Settings")
    retrieval_k = st.slider("📊 Top K", 1, 10, DEFAULT_CONFIG["retrieval_k"])
    search_type = st.selectbox("Search Type", ["mmr", "similarity", "similarity_score_threshold"])
    similarity_threshold = st.slider("📉 Threshold", 0.0, 1.0, 0.3) if search_type == "similarity_score_threshold" else 0.3

    # Advanced Features
    st.divider()
    st.subheader("🚀 Advanced Features")
    use_query_expansion = st.checkbox("🔎 Query Expansion", value=False)
    use_reranking = st.checkbox("🎯 Reranking (slower)", value=False)
    use_hybrid_search = st.checkbox("🔄 Hybrid Search (BM25 + Vector)", value=False)
    enable_pii_redaction = st.checkbox("🛡️ PII Redaction", value=False)

    # RAG Architectures
    st.divider()
    st.subheader("🗂️ RAG Architectures")
    st.caption("⚠️ Only ONE architecture will be active per query (priority order)")
    enable_memory = st.checkbox("💭 Conversation Memory", value=True)
    enable_adaptive = st.checkbox("🧩 Adaptive RAG", value=False)
    enable_corrective = st.checkbox("✅ Corrective RAG (CRAG)", value=False)
    enable_self_rag = st.checkbox("🔄 Self-RAG", value=False)
    enable_hyde = st.checkbox("💡 HyDe", value=False)
    enable_branched = st.checkbox("🌳 Branched RAG", value=False)
    enable_agentic = st.checkbox("🤖 Agentic RAG (slower but thorough)", value=False)

    # Advanced Retrieval & Agentic Refinement
    st.divider()
    st.subheader("⚡ Advanced Retrieval & Agentic Refinement")
    enable_query_caching = st.checkbox("💾 Query Caching", value=True)
    enable_query_decomposition = st.checkbox("🔍 Query Decomposition (for Agentic)", value=False)

    # Prompt Engineering
    st.divider()
    st.subheader("📝 Prompt Engineering")
    use_dynamic_prompting = st.checkbox("🔄 Dynamic Prompting", value=True)

    # Evaluation
    st.divider()
    enable_evaluation = st.checkbox("🧪 Enable Advanced Evaluation", value=False)

    # Build config
    config = {
        **DEFAULT_CONFIG,
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "chunking_method": chunking_method,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "use_hierarchical_retrieval": use_hierarchical_retrieval,
        "proposition_extraction": proposition_extraction,
        "multimodal_enabled": multimodal_enabled,
        "retrieval_k": retrieval_k,
        "search_type": search_type,
        "similarity_threshold": similarity_threshold,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "ocr_enabled": ocr_enabled,
        "extract_tables": extract_tables,
        "use_reranking": use_reranking,
        "use_query_expansion": use_query_expansion,
        "use_hybrid_search": use_hybrid_search,
        "enable_pii_redaction": enable_pii_redaction,
        "storage_path": storage_path,
        "enable_evaluation": enable_evaluation,
        "enable_memory": enable_memory,
        "enable_adaptive": enable_adaptive,
        "enable_corrective": enable_corrective,
        "enable_self_rag": enable_self_rag,
        "enable_hyde": enable_hyde,
        "enable_branched": enable_branched,
        "enable_agentic": enable_agentic,
        "enable_query_caching": enable_query_caching,
        "enable_query_decomposition": enable_query_decomposition,
        "use_dynamic_prompting": use_dynamic_prompting,
    }

    # Recalculate config hash if relevant settings change
    current_config_hash = hash_config_for_cache(config)
    if st.session_state.config_hash != current_config_hash:
        st.session_state.config_hash = current_config_hash

    # Ensure evaluator is initialized if evaluation is enabled
    if enable_evaluation and st.session_state.rag_evaluator is None:
        try:
            st.session_state.rag_evaluator = RAGEvaluator(config=EVALUATION_CONFIG)
            st.info("🧪 RAG Evaluator initialized for advanced metrics.")
        except Exception as e:
            st.error(f"Failed to initialize RAG Evaluator: {e}")
            logger.error(f"Evaluator init failed: {e}")

    # Batch Evaluation Button
    if st.button("🧪 Run Batch Evaluation"):
        if st.session_state.rag_evaluator is None:
            st.warning("🧪 Evaluation is disabled or not initialized. Enable 'Advanced Evaluation' in the sidebar.")
        elif "all_documents" not in st.session_state or not st.session_state.all_documents:
            st.warning("Load documents first.")
        else:
            try:
                test_cases = [
                    {
                        "query": "Sample query about document content",
                        "contexts": [doc["content"] for doc in st.session_state.all_documents[:5]],
                        "answer": None
                    }
                ]
                with st.spinner("Running batch evaluation..."):
                    batch_result = st.session_state.rag_evaluator.run_batch_evaluation(test_cases)
                    st.json(batch_result)
            except Exception as e:
                st.error(f"Batch evaluation failed: {e}")
                logger.error(f"Batch eval error: {e}")

    # Reset
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in st.session_state.keys():
            if key == "chat_history":
                st.session_state[key] = []
            elif key == "memory_manager":
                st.session_state[key].clear()
            else:
                st.session_state[key] = None
        st.session_state.config_hash = hash_config_for_cache(config)
        st.rerun()

# ==================== MAIN LAYOUT ====================
st.write("Debug: Testing column rendering")
col1, col2 = st.columns([1, 2])

# ==================== LEFT COLUMN: PDF LOADING ====================
with col1:
    st.write("Debug: Left column rendering")
    st.subheader("📄 Load PDFs")
    pdf_folder = st.text_input("📁 PDF Folder (optional)")
    uploaded_files = st.file_uploader("📤 Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("🚀 Load & Index", use_container_width=True):
        pdf_paths: List[str] = []
        if pdf_folder and os.path.isdir(pdf_folder):
            pdf_paths.extend([str(p) for p in Path(pdf_folder).glob("*.pdf")])
        if uploaded_files:
            temp_dir = tempfile.mkdtemp()
            for uf in uploaded_files:
                path = os.path.join(temp_dir, uf.name)
                with open(path, "wb") as f:
                    f.write(uf.getbuffer())
                pdf_paths.append(path)

        if not pdf_paths:
            st.error("❌ No PDFs provided!")
        else:
            with st.spinner("⏳ Processing..."):
                try:
                    chunks = docling_chunks_from_pdfs(config, pdf_paths)
                    if not chunks:
                        st.error("❌ No text extracted. Try enabling OCR.")
                    else:
                        chain, _, base_retriever, vectorstore, llm = setup_rag_chain(
                            config, chunks, st.session_state.memory_manager
                        )
                        st.session_state.update({
                            "rag_chain": chain,
                            "base_retriever": base_retriever,
                            "vectorstore": vectorstore,
                            "llm": llm,
                            "documents_loaded": True,
                            "all_documents": chunks
                        })

                        # Initialize advanced retrievers
                        if config["enable_adaptive"]:
                            st.session_state.adaptive_retriever = AdaptiveRetriever(
                                base_retriever, llm, config
                            )
                        if config["enable_corrective"]:
                            st.session_state.corrective_rag = CorrectiveRAG(
                                base_retriever, llm, config
                            )
                        if config["enable_self_rag"]:
                            st.session_state.self_rag = SelfRAG(
                                base_retriever, llm, config
                            )
                        if config["enable_hyde"]:
                            st.session_state.hyde_retriever = HyDERetriever(
                                base_retriever, llm, config
                            )
                        if config["enable_branched"]:
                            st.session_state.branched_rag = BranchedRAG(
                                base_retriever, llm, config
                            )
                        if config["enable_agentic"]:
                            st.session_state.agentic_rag = MetaAgent(
                                base_retriever, llm, config
                            )
                        st.success(f"✅ Indexed {len(chunks)} chunks!")
                except Exception as e:
                    st.error(f"❌ Indexing failed: {e}")
                    logger.exception("Indexing error")

# ==================== RIGHT COLUMN: CHAT INTERFACE ====================
with col2:
    st.write("Debug: Right column rendering")
    st.subheader("💬 Chat")
    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.memory_manager.clear()
        st.success("Chat history and conversation memory reset!")
        st.rerun()

    if not isinstance(st.session_state.chat_history, list):
        st.error("Chat history is corrupted. Resetting...")
        st.session_state.chat_history = []

    try:
        for idx, msg in enumerate(st.session_state.chat_history):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant":
                    with st.expander("📜 Sources", expanded=False):
                        for src in msg.get("sources", []):
                            st.write(f"**{src['metadata'].get('filename', 'Unknown')}** (Page {src['metadata'].get('page', 'N/A')}):")
                            st.write(src["content"][:200] + ("..." if len(src["content"]) > 200 else ""))
                    if eval_data := msg.get("evaluation"):
                        with st.expander("🧪 Evaluation Metrics", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Faithfulness:** {eval_data.get('faithfulness', 'N/A')} (Score: {eval_data.get('faithfulness', 0):.2f})")
                                st.write(f"**Relevance:** {eval_data.get('relevance', 'N/A')} (Score: {eval_data.get('relevance', 0):.2f})")
                                st.write(f"**Hallucination:** {'Detected' if eval_data.get('has_hallucination', False) else 'None'} (Severity: {eval_data.get('hallucination_severity', 'none')})")
                            with col2:
                                st.write(f"**Context Precision:** {eval_data.get('context_precision', 0) * 100:.1f}%")
                                st.write(f"**Failure Category:** {eval_data.get('failure_category', 'None')}")
                                st.write(f"**Eval Latency:** {eval_data.get('eval_latency_ms', 'N/A')}ms")
                            if config["enable_evaluation"] and st.session_state.rag_evaluator:
                                session_stats = st.session_state.rag_evaluator.get_session_stats()
                                st.write(f"**Session Queries:** {session_stats['total_queries']}")
                                st.write(f"**Cache Hit Rate:** {session_stats.get('cache_hit_rate', 0):.1%}")
                                st.write(f"**Failure Rate:** {session_stats.get('failure_rate', 0):.1%}")

                    # Feedback
                    feedback_key = f"fb_{idx}"
                    cols = st.columns([1, 1, 8])
                    with cols[0]:
                        if st.button("👍", key=f"up_{feedback_key}"):
                            save_feedback(msg.get("question", ""), msg["content"], 1)
                            st.toast("Thank you!")
                    with cols[1]:
                        if st.button("👎", key=f"down_{feedback_key}"):
                            save_feedback(msg.get("question", ""), msg["content"], -1)
                            st.toast("Thank you!")
    except Exception as e:
        st.error(f"Error rendering chat history: {e}")
        logger.error(f"Chat history rendering error: {e}")

    if "all_documents" not in st.session_state or not st.session_state.all_documents:
        st.warning("No documents loaded. Upload PDFs or specify a folder and click 'Load & Index'.")
    if prompt := st.chat_input("Ask about your PDFs..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("🤔 Thinking..."):
            try:
                start_time = time.time()
                msg_data = {"question": prompt}

                # Check Cache First if enabled
                cache_hit = False
                answer = None
                sources = []
                
                if config["enable_query_caching"]:
                    cached_result = get_cached_result(prompt, st.session_state.config_hash)
                    if cached_result:
                        answer, sources = cached_result
                        cache_hit = True
                        st.info("⚡ Retrieved from cache!")

                if not cache_hit:
                    retrieval_method_used = "Base"
                    
                    if config["enable_agentic"]:
                        with st.spinner("🤖 Consulting document agents..."):
                            answer = st.session_state.agentic_rag.synthesize_answer(prompt)
                        sources = st.session_state.base_retriever.invoke(prompt)
                        retrieval_method_used = "Agentic (Meta-Agent)"
                        
                    elif config["enable_branched"]:
                        sources, domain = st.session_state.branched_rag.retrieve_branched(prompt)
                        retrieval_method_used = f"Branched ({domain})"
                        
                    elif config["enable_self_rag"]:
                        context_text = "\n".join([doc.page_content for doc in 
                                                st.session_state.base_retriever.invoke(prompt)[:3]])
                        self_rag_result = st.session_state.self_rag.generate_with_self_retrieval(
                            prompt, context_text
                        )
                        answer = self_rag_result["final_answer"]
                        sources = st.session_state.base_retriever.invoke(prompt)
                        msg_data["self_rag_data"] = {
                            "iterations": self_rag_result["total_iterations"],
                            "iteration_details": self_rag_result["iterations"]
                        }
                        retrieval_method_used = "Self-RAG"
                        
                    elif config["enable_hyde"]:
                        sources, hypothetical = st.session_state.hyde_retriever.retrieve_with_hyde(prompt)
                        msg_data["hyde_hypothetical"] = hypothetical[:300]
                        retrieval_method_used = "HyDe"
                        
                    elif config["enable_corrective"]:
                        initial_sources = st.session_state.base_retriever.invoke(prompt)
                        sources = st.session_state.corrective_rag.correct_retrieval(
                            prompt, initial_sources
                        )
                        coverage = len([s for s in sources if s]) / len(initial_sources) if initial_sources else 0
                        msg_data["corrective_info"] = {"coverage": coverage}
                        retrieval_method_used = "Corrective"
                        
                    elif config["enable_adaptive"]:
                        adaptive_info = {
                            "complexity": st.session_state.adaptive_retriever.classify_query_complexity(prompt)
                        }
                        sources = st.session_state.adaptive_retriever.retrieve_adaptive(
                            prompt, config["retrieval_k"]
                        )
                        msg_data["adaptive_info"] = adaptive_info
                        retrieval_method_used = "Adaptive"
                        
                    else:
                        sources = st.session_state.base_retriever.invoke(prompt)
                        retrieval_method_used = "Base"

                    if answer is None:
                        query = expand_query(prompt, st.session_state.llm) if config["use_query_expansion"] else prompt
                        answer = st.session_state.rag_chain.invoke(query)

                    if config["enable_query_caching"]:
                        cache_result(prompt, st.session_state.config_hash, answer, sources)

                formatted_sources = [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in sources
                ]

                evaluation = {}
                if config["enable_evaluation"] and st.session_state.rag_evaluator:
                    with st.spinner("🧪 Evaluating..."):
                        try:
                            evaluator = st.session_state.rag_evaluator
                            metadata = {
                                "latency_ms": (time.time() - start_time) * 1000,
                                "retrieval_method_used": retrieval_method_used,
                                "config_features": [k for k, v in config.items() if v is True and k.startswith(('enable_', 'use_'))]
                            }
                            eval_result = evaluator.evaluate_response(
                                query=prompt,
                                answer=answer,
                                retrieved_contexts=[doc["content"] for doc in formatted_sources],
                                metadata=metadata
                            )
                            evaluation = eval_result
                            evaluator.track_component_latency("retrieval", metadata.get("retrieval_latency_ms", 0))
                            evaluator.track_component_latency("generation", metadata.get("generation_latency_ms", 0))
                            evaluator.track_cache_hit(cache_hit)
                            alerts = evaluator.check_alerts()
                            for alert in alerts:
                                st.warning(f"⚠️ {alert}")
                        except Exception as e:
                            st.warning(f"Advanced evaluation failed: {e}. Falling back to basic.")
                            embeddings = create_embeddings(config["embedding_model"])
                            evaluation = evaluate_rag_response_full(
                                question=prompt,
                                answer=answer,
                                contexts=[doc["content"] for doc in formatted_sources],
                                llm=st.session_state.llm,
                                embeddings=embeddings,
                                start_time=start_time
                            )

                msg_data.update({
                    "role": "assistant",
                    "content": answer,
                    "sources": formatted_sources,
                    "evaluation": evaluation,
                })
                st.session_state.chat_history.append(msg_data)

                log_interaction({
                    "question": prompt,
                    "answer": answer,
                    "sources": [s["metadata"].get("filename") for s in formatted_sources],
                    "evaluation": evaluation,
                    "timestamp": time.time(),
                    "config_features": [
                        k for k, v in config.items() 
                        if v is True and k.startswith(('enable_', 'use_'))
                    ]
                })

                if config["enable_memory"]:
                    st.session_state.memory_manager.add_turn(
                        prompt, answer, lambda x: len(x.split())
                    )

                st.rerun()

            except Exception as e:
                st.error(f"❌ Response error: {e}")
                logger.exception("Chat error")
    else:
        st.info("👈 Load PDFs to start chatting!")
        st.markdown("""
        ### 🚀 Quick Start
        1. Make sure **Ollama is running**
        2. Upload PDFs or enter a folder path
        3. Click **Load & Index**
        4. Ask questions!
        
        ### 🗂️ Available Architectures
        - **Simple RAG:** Basic retrieval + generation
        - **Memory:** Remember conversation context
        - **Adaptive:** Smart query complexity routing
        - **Corrective:** Validate retrieved documents
        - **Self-RAG:** Iterative self-critique
        - **HyDe:** Hypothetical document guidance
        - **Branched:** Route to specialized knowledge bases
        - **Agentic:** Use document-specific agents (thorough but slower)
        
        ### ⚡ Performance Tips
        - Enable **Caching** for faster repeated queries
        - Use **Dynamic Prompting** for context-aware responses
        - **Agentic RAG** is most thorough but takes longer
        - Only one architecture runs per query (priority order)
        """)