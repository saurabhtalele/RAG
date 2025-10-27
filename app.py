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
    SelfRAG, HyDERetriever, BranchedRAG, MetaAgent,
    get_cached_result, cache_result, hash_config_for_cache
)
from rag_evaluator import RAGEvaluator

# ==================== SESSION STATE INITIALIZATION ====================
for key in ["chat_history", "rag_chain", "base_retriever", "vectorstore", "llm",
            "documents_loaded", "memory_manager", "adaptive_retriever",
            "corrective_rag", "self_rag", "hyde_retriever", "branched_rag",
            "agentic_rag", "rag_evaluator", "graph_retriever", "all_chunks",
            "retriever_for_chain"]:
    if key not in st.session_state:
        if key == "chat_history":
            st.session_state[key] = []
        elif key == "memory_manager":
            st.session_state[key] = ConversationMemoryManager(max_memory_tokens=2000)
        else:
            st.session_state[key] = None

if "config_hash" not in st.session_state:
    st.session_state.config_hash = hash_config_for_cache(DEFAULT_CONFIG)

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="📚 Advanced RAG Playground", page_icon="📚", layout="wide")
st.title("📚 Advanced RAG Playground")
st.caption("Local AI + Smart PDFs + 8 RAG Architectures + New Features – All on your machine!")

# ==================== SIDEBAR CONFIGURATION ====================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Initialize evaluator
    if st.session_state.rag_evaluator is None and DEFAULT_CONFIG["enable_evaluation"]:
        try:
            st.session_state.rag_evaluator = RAGEvaluator(config=EVALUATION_CONFIG)
            st.info("🧪 RAG Evaluator initialized")
        except Exception as e:
            st.error(f"Evaluator init failed: {e}")
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
    
    # NEW FEATURES
    st.divider()
    st.subheader("✨ New Advanced Features")
    enable_knowledge_graph = st.checkbox("🧠 Knowledge Graph Retrieval", value=False)
    enable_metadata_filtering = st.checkbox("🔍 Metadata Filtering (page:X, file:Y)", value=True)
    use_rrf_fusion = st.checkbox("🔄 RRF Multi-Retriever Fusion", value=False)
    
    # RAG Architectures
    st.divider()
    st.subheader("🗂️ RAG Architectures")
    st.caption("⚠️ Only ONE architecture active per query (priority order)")
    enable_memory = st.checkbox("💭 Conversation Memory", value=True)
    enable_adaptive = st.checkbox("🧩 Adaptive RAG", value=False)
    enable_corrective = st.checkbox("✅ Corrective RAG (CRAG)", value=False)
    enable_self_rag = st.checkbox("🔄 Self-RAG", value=False)
    enable_hyde = st.checkbox("💡 HyDe", value=False)
    enable_branched = st.checkbox("🌳 Branched RAG", value=False)
    enable_agentic = st.checkbox("🤖 Agentic RAG (slower but thorough)", value=False)
    
    # Advanced Options
    st.divider()
    st.subheader("⚡ Advanced Options")
    enable_query_caching = st.checkbox("💾 Query Caching", value=True)
    enable_query_decomposition = st.checkbox("🔍 Query Decomposition", value=False)
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
        # New features
        "enable_knowledge_graph": enable_knowledge_graph,
        "enable_metadata_filtering": enable_metadata_filtering,
        "use_rrf_fusion": use_rrf_fusion,
    }
    
    # Update config hash
    current_config_hash = hash_config_for_cache(config)
    if st.session_state.config_hash != current_config_hash:
        st.session_state.config_hash = current_config_hash
    
    # Ensure evaluator initialized if needed
    if enable_evaluation and st.session_state.rag_evaluator is None:
        try:
            st.session_state.rag_evaluator = RAGEvaluator(config=EVALUATION_CONFIG)
            st.info("🧪 RAG Evaluator initialized")
        except Exception as e:
            st.error(f"Evaluator init failed: {e}")
            logger.error(f"Evaluator init failed: {e}")

# ==================== MAIN CONTENT ====================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📂 Document Loading")
    
    # File Upload
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files"
    )
    
    # OR Folder Path
    st.write("OR")
    folder_path = st.text_input(
        "📁 Enter folder path containing PDFs",
        placeholder="C:/Documents/PDFs or /home/user/pdfs"
    )
    
    if st.button("🚀 Load & Index Documents", type="primary"):
        pdf_paths = []
        
        # Handle uploaded files
        if uploaded_files:
            temp_dir = tempfile.mkdtemp()
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                pdf_paths.append(file_path)
        
        # Handle folder path
        if folder_path and os.path.isdir(folder_path):
            folder_pdfs = list(Path(folder_path).glob("*.pdf"))
            pdf_paths.extend([str(p) for p in folder_pdfs])
        
        if not pdf_paths:
            st.error("⚠️ No PDFs found! Upload files or provide a valid folder path.")
        else:
            with st.spinner(f"📖 Processing {len(pdf_paths)} PDFs..."):
                try:
                    # Extract chunks
                    chunks = docling_chunks_from_pdfs(config, pdf_paths)
                    
                    if not chunks:
                        st.error("❌ No text extracted from PDFs. Check OCR settings or file quality.")
                    else:
                        st.success(f"✅ Extracted {len(chunks)} chunks from {len(pdf_paths)} PDFs")
                        
                        # Store chunks
                        st.session_state.all_chunks = chunks
                        
                        # Setup RAG pipeline
                        with st.spinner("🔧 Building RAG pipeline..."):
                            result = setup_rag_chain(config, chunks, st.session_state.memory_manager)
                            
                            # Unpack results
                            if len(result) == 6:
                                chain, retriever_for_chain, base_retriever, vectorstore, llm, graph_retriever = result
                                st.session_state.graph_retriever = graph_retriever
                            else:
                                chain, retriever_for_chain, base_retriever, vectorstore, llm = result
                                st.session_state.graph_retriever = None
                            
                            st.session_state.rag_chain = chain
                            st.session_state.retriever_for_chain = retriever_for_chain
                            st.session_state.base_retriever = base_retriever
                            st.session_state.vectorstore = vectorstore
                            st.session_state.llm = llm
                            st.session_state.documents_loaded = True
                            
                            # Initialize architecture components
                            st.session_state.adaptive_retriever = AdaptiveRetriever(base_retriever, llm, config)
                            st.session_state.corrective_rag = CorrectiveRAG(base_retriever, llm, config)
                            st.session_state.self_rag = SelfRAG(base_retriever, llm, config)
                            st.session_state.hyde_retriever = HyDERetriever(base_retriever, llm, config)
                            st.session_state.branched_rag = BranchedRAG(base_retriever, llm, config)
                            st.session_state.agentic_rag = MetaAgent(base_retriever, llm, config)
                            
                            st.success("✅ RAG pipeline ready!")
                            st.info(f"💾 Vectorstore: {len(chunks)} embeddings indexed")
                            
                            if st.session_state.graph_retriever:
                                st.info("🧠 Knowledge Graph: Entities extracted")
                
                except Exception as e:
                    st.error(f"❌ Failed to build RAG pipeline: {e}")
                    logger.exception("Pipeline build error")

with col2:
    st.subheader("💬 Chat Interface")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                
                # Show sources
                if "sources" in msg:
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(msg["sources"][:3], 1):
                            st.markdown(f"**Source {i}:** {src['metadata'].get('filename', 'Unknown')}")
                            st.caption(f"Page: {src['metadata'].get('page', 'N/A')}")
                            st.text(src["content"][:200] + "...")
                
                # Show evaluation
                if "evaluation" in msg and msg["evaluation"]:
                    with st.expander("🧪 Evaluation Metrics"):
                        eval_data = msg["evaluation"]
                        cols = st.columns(3)
                        
                        if "faithfulness" in eval_data:
                            cols[0].metric("Faithfulness", 
                                f"{eval_data['faithfulness']:.2f}" if isinstance(eval_data['faithfulness'], float) 
                                else eval_data['faithfulness'])
                        if "relevance" in eval_data:
                            cols[1].metric("Relevance", 
                                f"{eval_data['relevance']:.2f}" if isinstance(eval_data['relevance'], float) 
                                else eval_data['relevance'])
                        if "context_precision" in eval_data:
                            cols[2].metric("Context Precision", f"{eval_data['context_precision']:.2f}")
                        
                        if eval_data.get("has_hallucination"):
                            st.warning(f"⚠️ Hallucination: {eval_data.get('hallucinated_claims', 'N/A')}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.documents_loaded:
            st.warning("👈 Please load documents first!")
        else:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        start_time = time.time()
                        msg_data = {}
                        
                        # Check cache
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
                            
                            # Architecture routing
                            if config["enable_agentic"]:
                                with st.spinner("🤖 Consulting document agents..."):
                                    answer = st.session_state.agentic_rag.synthesize_answer(prompt)
                                    sources = st.session_state.base_retriever.invoke(prompt)
                                    retrieval_method_used = "Agentic"
                            
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
                                    "iterations": self_rag_result["total_iterations"]
                                }
                                retrieval_method_used = "Self-RAG"
                            
                            elif config["enable_hyde"]:
                                sources, hypothetical = st.session_state.hyde_retriever.retrieve_with_hyde(prompt)
                                msg_data["hyde_hypothetical"] = hypothetical[:300]
                                retrieval_method_used = "HyDe"
                            
                            elif config["enable_corrective"]:
                                initial_sources = st.session_state.base_retriever.invoke(prompt)
                                sources = st.session_state.corrective_rag.correct_retrieval(prompt, initial_sources)
                                retrieval_method_used = "Corrective"
                            
                            elif config["enable_adaptive"]:
                                sources = st.session_state.adaptive_retriever.retrieve_adaptive(
                                    prompt, config["retrieval_k"]
                                )
                                retrieval_method_used = "Adaptive"
                            
                            else:
                                sources = st.session_state.base_retriever.invoke(prompt)
                                retrieval_method_used = "Base"
                            
                            # Generate answer if not already generated
                            if answer is None:
                                query = expand_query(prompt, st.session_state.llm) if config["use_query_expansion"] else prompt
                                answer = st.session_state.rag_chain.invoke(query)
                            
                            # Cache result
                            if config["enable_query_caching"]:
                                cache_result(prompt, st.session_state.config_hash, answer, sources)
                        
                        # Format sources
                        formatted_sources = [
                            {"content": doc.page_content, "metadata": doc.metadata}
                            for doc in sources
                        ]
                        
                        # Evaluation
                        evaluation = {}
                        if config["enable_evaluation"] and st.session_state.rag_evaluator:
                            with st.spinner("🧪 Evaluating..."):
                                try:
                                    evaluator = st.session_state.rag_evaluator
                                    metadata = {
                                        "latency_ms": (time.time() - start_time) * 1000,
                                        "retrieval_method_used": retrieval_method_used if not cache_hit else "Cache"
                                    }
                                    
                                    eval_result = evaluator.evaluate_response(
                                        query=prompt,
                                        answer=answer,
                                        retrieved_contexts=[doc["content"] for doc in formatted_sources],
                                        metadata=metadata
                                    )
                                    evaluation = eval_result
                                    evaluator.track_cache_hit(cache_hit)
                                
                                except Exception as e:
                                    st.warning(f"Advanced evaluation failed: {e}")
                                    embeddings = create_embeddings(config["embedding_model"])
                                    evaluation = evaluate_rag_response_full(
                                        question=prompt,
                                        answer=answer,
                                        contexts=[doc["content"] for doc in formatted_sources],
                                        llm=st.session_state.llm,
                                        embeddings=embeddings,
                                        start_time=start_time
                                    )
                        
                        # Display answer
                        st.write(answer)
                        
                        # Update message
                        msg_data.update({
                            "role": "assistant",
                            "content": answer,
                            "sources": formatted_sources,
                            "evaluation": evaluation,
                        })
                        
                        st.session_state.chat_history.append(msg_data)
                        
                        # Log
                        log_interaction({
                            "question": prompt,
                            "answer": answer,
                            "sources": [s["metadata"].get("filename") for s in formatted_sources],
                            "timestamp": time.time()
                        })
                        
                        # Update memory
                        if config["enable_memory"]:
                            st.session_state.memory_manager.add_turn(
                                prompt, answer, lambda x: len(x.split())
                            )
                        
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Response error: {e}")
                        logger.exception("Chat error")
    
    # Info when no documents
    if not st.session_state.documents_loaded:
        st.info("👈 Load PDFs to start chatting!")
        st.markdown("""
### 🚀 Quick Start
1. **Ollama must be running**: `ollama serve`
2. Upload PDFs or enter folder path
3. Click **Load & Index**
4. Ask questions!

### 🗂️ RAG Architectures
- **Simple RAG**: Basic retrieval + generation
- **Memory**: Conversation context
- **Adaptive**: Smart complexity routing
- **Corrective**: Validates retrieval quality
- **Self-RAG**: Iterative self-critique
- **HyDe**: Hypothetical document guidance
- **Branched**: Domain-specific routing
- **Agentic**: Document-specific agents

### ✨ New Features
- **Knowledge Graph**: Entity-based retrieval
- **Metadata Filtering**: `page:5-10`, `file:report.pdf`
- **RRF Fusion**: Multi-retriever combination

### ⚡ Tips
- Enable **Caching** for speed
- **Dynamic Prompting** for better context
- One architecture per query (priority order)
""")

# ==================== FOOTER ====================
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.documents_loaded:
        st.metric("Documents", "✅ Ready")
    else:
        st.metric("Documents", "❌ None")

with col2:
    st.metric("Messages", len(st.session_state.chat_history))

with col3:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        if st.session_state.memory_manager:
            st.session_state.memory_manager.clear()
        st.rerun()
