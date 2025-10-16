"""
MAIN RAG APPLICATION
Save this as: rag_app.py
Run with: streamlit run rag_app.py
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dependencies with error handling
try:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.chat_models import ChatOllama
    from langchain_chroma import Chroma
    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
except ImportError as e:
    st.error(f"❌ LangChain import error: {e}")
    st.info("Please run: pip install langchain langchain-community langchain-chroma")
    st.stop()

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.chunking import HybridChunker, HierarchicalChunker
except ImportError as e:
    st.error(f"❌ Docling import error: {e}")
    st.info("Please run: pip install docling")
    st.stop()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sanitize_metadata(md: dict, max_str_len: int = 10000) -> dict:
    """Sanitize metadata to ensure it's JSON-serializable and not too large"""
    def to_jsonable(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        try:
            s = str(v)
            return s if len(s) <= max_str_len else s[:max_str_len] + "..."
        except Exception:
            return None
    
    out = {}
    for k, v in (md or {}).items():
        try:
            out[str(k)] = to_jsonable(v)
        except Exception:
            out[str(k)] = None
    return out

def check_ollama_connection():
    """Check if Ollama is running"""
    try:
        test_embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        test_embeddings.embed_query("test")
        return True
    except Exception as e:
        logger.error(f"Ollama connection error: {e}")
        return False

def expand_query(query: str, llm) -> str:
    """Expand query with related terms for better retrieval"""
    expansion_prompt = f"""Given this question: "{query}"

Generate 2-3 related search queries that would help find relevant information.
Format as comma-separated queries.

Related queries:"""
    
    try:
        expanded = llm.invoke(expansion_prompt)
        return f"{query} {expanded}"
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return query

def create_reranked_retriever(base_retriever, llm):
    """Add reranking to improve result quality"""
    try:
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        return compression_retriever
    except Exception as e:
        logger.warning(f"Reranking setup failed, using base retriever: {e}")
        return base_retriever

def get_existing_collections(storage_path):
    """Get list of existing ChromaDB collections"""
    try:
        if not os.path.exists(storage_path):
            return []
        
        # Simple check for chroma.sqlite3 file
        if os.path.exists(os.path.join(storage_path, "chroma.sqlite3")):
            return ["existing_collection"]
        return []
    except Exception as e:
        logger.error(f"Error checking existing collections: {e}")
        return []

def rag_chain_from_params(params, all_chunks):
    """Sets up vectorstore, retriever, prompt and the full rag chain based on config and given chunks"""
    try:
        # Validate chunks
        if not all_chunks:
            raise ValueError("No chunks provided. Please check your PDF processing.")
        
        # Filter out empty chunks
        valid_chunks = [
            chunk for chunk in all_chunks 
            if chunk.get("content") and chunk["content"].strip()
        ]
        
        if not valid_chunks:
            raise ValueError("All chunks are empty. Please check your PDF content and processing settings.")
        
        st.write(f"✓ Validated {len(valid_chunks)} non-empty chunks (filtered {len(all_chunks) - len(valid_chunks)} empty chunks)")
        
        st.write("🔧 Initializing embedding model...")
        # Init embedding model
        embeddings = OllamaEmbeddings(
            model=params["embedding_model"],
            show_progress=True
        )
        
        st.write("📚 Creating optimized vector store...")
        
        # Determine storage directory
        if params.get("storage_path"):
            persist_directory = params["storage_path"]
            os.makedirs(persist_directory, exist_ok=True)
            st.write(f"📁 Using persistent storage: {persist_directory}")
        else:
            persist_directory = tempfile.mkdtemp()
            st.write(f"📁 Using temporary storage (will be deleted on session end)")
        
        # Create ChromaDB with optimized settings
        texts = [chunk["content"] for chunk in valid_chunks]
        metadatas = [chunk["metadata"] for chunk in valid_chunks]
        collection_name = f"rag_collection_{uuid.uuid4().hex[:8]}"
        
        vector_store = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            collection_name=collection_name,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}  # Cosine similarity for better results
        )
        
        st.write(f"✓ Created collection: {collection_name}")
        
        st.write("🔍 Setting up enhanced retriever...")
        # Enhanced retriever configuration based on search type
        if params["search_type"] == "similarity_score_threshold":
            retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": params["retrieval_k"],
                    "score_threshold": params["similarity_threshold"]
                }
            )
        elif params["search_type"] == "mmr":
            # MMR for diversity and relevance balance
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": params["retrieval_k"],
                    "fetch_k": params["retrieval_k"] * 3,  # Fetch more candidates
                    "lambda_mult": 0.5  # Balance between relevance (1.0) and diversity (0.0)
                }
            )
        else:  # similarity
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": params["retrieval_k"]
                }
            )
        
        st.write("🤖 Loading LLM...")
        # LLM setup
        llm = ChatOllama(
            model=params["llm_model"],
            temperature=params["temperature"],
            num_predict=params["max_tokens"]
        )
        
        # Add reranking if enabled
        if params.get("use_reranking", False):
            st.write("🎯 Adding reranking layer...")
            retriever = create_reranked_retriever(retriever, llm)
        
        st.write("⚙️ Building chain...")
        # Enhanced prompt template
        prompt_template = """You are a helpful assistant for question-answering tasks. 
Use the following context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise and accurate.

Context: {context}
Question: {question}
Answer:"""
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain, retriever, vector_store, llm
    
    except Exception as e:
        st.error(f"❌ Error setting up RAG chain: {e}")
        logger.exception(f"RAG chain setup error: {e}")
        raise

def docling_chunks_from_pdfs(params, pdf_paths):
    """Loads PDFs using Docling and returns all chunks"""
    try:
        st.write("📋 Initializing document converter...")
        
        # Try new API first, fallback to old API
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = params["ocr_enabled"]
            pipeline_options.do_table_structure = params["extract_tables"]
            
            doc_converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                pipeline_options=pipeline_options
            )
        except TypeError:
            # Fallback for older Docling versions
            st.write("  ├─ Using legacy API...")
            doc_converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF]
            )
        
        all_chunks = []
        total_pdfs = len(pdf_paths)
        
        for idx, pdf_path in enumerate(pdf_paths, 1):
            try:
                st.write(f"📄 Processing PDF {idx}/{total_pdfs}: {Path(pdf_path).name}...")
                
                result = doc_converter.convert(pdf_path)
                doc = result.document
                
                # Chunking selection
                if params["chunking_strategy"] == "hierarchical":
                    st.write(f"  ├─ Using hierarchical chunking...")
                    chunker = HierarchicalChunker(
                        max_tokens=params["chunk_size"],
                        overlap=params["chunk_overlap"]
                    )
                    chunk_iter = chunker.chunk(doc)
                
                elif params["chunking_strategy"] == "hybrid":
                    st.write(f"  ├─ Using hybrid chunking...")
                    chunker = HybridChunker(
                        max_tokens=params["chunk_size"],
                        overlap=params["chunk_overlap"]
                    )
                    chunk_iter = chunker.chunk(doc)
                
                else:  # fixed
                    st.write(f"  ├─ Using fixed-size chunking...")
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    markdown_text = doc.export_to_markdown()
                    
                    # Check if markdown is empty
                    if not markdown_text or not markdown_text.strip():
                        st.warning(f"  └─ ⚠️ No text extracted from {Path(pdf_path).name}")
                        continue
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=params["chunk_size"],
                        chunk_overlap=params["chunk_overlap"],
                        length_function=len,
                    )
                    chunks = text_splitter.split_text(markdown_text)
                    
                    for chunk in chunks:
                        if chunk and chunk.strip():  # Only add non-empty chunks
                            metadata = {
                                "source": str(pdf_path),
                                "filename": Path(pdf_path).name
                            }
                            metadata = sanitize_metadata(metadata)
                            
                            all_chunks.append({
                                "content": chunk,
                                "metadata": metadata
                            })
                    
                    st.write(f"  └─ ✓ Extracted {len(chunks)} chunks")
                    continue
                
                # Process hierarchical/hybrid chunks
                chunk_count = 0
                for chunk in chunk_iter:
                    if chunk.text and chunk.text.strip():
                        # Normalize Docling metadata into a dict safely
                        meta_dict = {}
                        try:
                            if hasattr(chunk, "meta") and chunk.meta is not None:
                                # Try to_dict() method first
                                if hasattr(chunk.meta, "to_dict") and callable(chunk.meta.to_dict):
                                    meta_dict = chunk.meta.to_dict()
                                else:
                                    # Try mapping conversion; if it fails, stringify
                                    try:
                                        meta_dict = dict(chunk.meta)
                                    except Exception:
                                        meta_dict = {"meta": str(chunk.meta)}
                        except Exception as e:
                            logger.warning(f"Could not extract chunk metadata: {e}")
                            meta_dict = {}
                        
                        # Build metadata without using ** on non-mapping
                        metadata = {
                            "source": str(pdf_path),
                            "filename": Path(pdf_path).name,
                            "page": getattr(chunk, "page_num", None),
                        }
                        
                        # Merge safely
                        if isinstance(meta_dict, dict):
                            metadata.update(meta_dict)
                        
                        # Sanitize metadata
                        metadata = sanitize_metadata(metadata)
                        
                        all_chunks.append({
                            "content": chunk.text,
                            "metadata": metadata,
                        })
                        chunk_count += 1
                
                st.write(f"  └─ ✓ Extracted {chunk_count} chunks")
            
            except Exception as e:
                st.warning(f"⚠️ Failed to process {Path(pdf_path).name}: {e}")
                logger.exception(f"PDF processing error for {pdf_path}: {e}")
                continue
        
        return all_chunks
    
    except Exception as e:
        st.error(f"❌ Error processing PDFs: {e}")
        logger.exception(f"PDF chunking error: {e}")
        raise

# ============================================================================
# STREAMLIT APP SETUP
# ============================================================================

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

st.set_page_config(
    page_title="PDF RAG Playground",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Playground")
st.caption("Docling + Ollama + ChromaDB + Streamlit (Enhanced Retrieval)")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check Ollama
    with st.expander("🔌 Connection Status", expanded=True):
        if st.button("Check Ollama Connection"):
            with st.spinner("Testing connection..."):
                if check_ollama_connection():
                    st.success("✅ Ollama is running and accessible!")
                else:
                    st.error("❌ Cannot connect to Ollama. Make sure it's running on http://localhost:11434")
    
    # Storage Configuration
    with st.expander("💾 Storage Settings", expanded=True):
        use_persistent_storage = st.checkbox(
            "Use Persistent Storage",
            value=True,
            help="Save ChromaDB to disk for reuse across sessions"
        )
        
        if use_persistent_storage:
            default_storage = os.path.join(os.getcwd(), "chromadb_storage")
            storage_path = st.text_input(
                "ChromaDB Storage Path",
                value=default_storage,
                help="Where to save the vector database"
            )
            
            # Show existing collections if any
            existing = get_existing_collections(storage_path)
            if existing and os.path.exists(storage_path):
                st.info(f"📂 Storage folder exists with {len(os.listdir(storage_path))} files")
        else:
            storage_path = None
            st.warning("⚠️ Temporary storage will be deleted when session ends")
    
    # LLM Configuration
    with st.expander("🤖 LLM Settings", expanded=True):
        llm_model = st.text_input(
            "LLM Model",
            value="gemma2:2b",
            help="E.g., gemma2:2b, llama2, neural-chat, etc."
        )
        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.7, 0.1,
            help="Higher = more creative, Lower = more focused"
        )
        max_tokens = st.slider(
            "Max Response Tokens",
            128, 2048, 512, 128
        )
    
    # Embedding Configuration
    with st.expander("🔗 Embedding Settings", expanded=True):
        embedding_model = st.text_input(
            "Embedding Model",
            value="nomic-embed-text:latest",
            help="Must be installed in Ollama"
        )
    
    # Chunking Configuration
    with st.expander("✂️ Chunking Settings", expanded=True):
        chunking_strategy = st.selectbox(
            "Chunking Strategy",
            ["hierarchical", "hybrid", "fixed"],
            index=0,
            help="hierarchical: structure-aware, hybrid: balanced, fixed: simple"
        )
        chunk_size = st.slider(
            "Chunk Size (tokens)",
            128, 2048, 800, 128,
            help="Larger chunks = more context, but less precision"
        )
        chunk_overlap = st.slider(
            "Chunk Overlap",
            0, 512, 100, 10,
            help="Overlap helps maintain context between chunks"
        )
    
    # Retrieval Configuration
    with st.expander("🔍 Retrieval Settings", expanded=True):
        retrieval_k = st.slider(
            "Top K Results",
            1, 10, 5,
            help="Number of chunks to retrieve"
        )
        search_type = st.selectbox(
            "Search Type",
            ["mmr", "similarity_score_threshold", "similarity"],
            index=0,
            help="MMR balances relevance and diversity"
        )
        if search_type == "similarity_score_threshold":
            similarity_threshold = st.slider(
                "Similarity Threshold",
                0.0, 1.0, 0.3, 0.05,
                help="Lower = more results, Higher = more strict"
            )
        else:
            similarity_threshold = 0.3
        
        use_reranking = st.checkbox(
            "Enable Reranking",
            value=False,
            help="Rerank results using LLM for better relevance (slower)"
        )
        
        use_query_expansion = st.checkbox(
            "Enable Query Expansion",
            value=False,
            help="Expand queries with related terms (slower but better)"
        )
    
    # Document Processing
    with st.expander("📄 Document Processing", expanded=True):
        ocr_enabled = st.checkbox(
            "Enable OCR",
            value=False,
            help="Requires Docling models (slower)"
        )
        extract_tables = st.checkbox(
            "Extract Tables",
            value=False,
            help="Requires Docling models (slower)"
        )
    
    # Compile config
    config = {
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunking_strategy": chunking_strategy,
        "retrieval_k": retrieval_k,
        "similarity_threshold": similarity_threshold,
        "search_type": search_type,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "ocr_enabled": ocr_enabled,
        "extract_tables": extract_tables,
        "use_reranking": use_reranking,
        "use_query_expansion": use_query_expansion,
        "storage_path": storage_path if use_persistent_storage else None,
    }
    
    with st.expander("📋 Current Configuration"):
        st.json(config)
    
    if st.button("🔄 Reset Session", type="secondary"):
        for key in ["chat_history", "rag_chain", "retriever", "vector_store", "llm", "documents_loaded"]:
            if key == "chat_history":
                st.session_state[key] = []
            elif key == "documents_loaded":
                st.session_state[key] = False
            else:
                st.session_state[key] = None
        st.rerun()

# ============================================================================
# MAIN LAYOUT
# ============================================================================

col1, col2 = st.columns([1, 2])

# LEFT COLUMN: Document Upload
with col1:
    st.subheader("📁 PDF Documents")
    
    pdf_folder = st.text_input(
        "PDF Folder Path (optional)",
        value="",
        help="Absolute path to folder with PDFs, e.g., C:\\Users\\YourName\\Documents\\PDFs"
    )
    
    uploaded_files = st.file_uploader(
        "Or Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.button("🚀 Load & Index PDFs", use_container_width=True):
        pdf_files = []
        
        # Collect PDFs from folder
        if pdf_folder and os.path.exists(pdf_folder):
            folder_pdfs = list(Path(pdf_folder).glob("*.pdf"))
            st.write(f"✓ Found {len(folder_pdfs)} PDFs in folder")
            pdf_files.extend([str(p) for p in folder_pdfs])
        
        # Collect uploaded PDFs
        if uploaded_files:
            st.write(f"✓ Found {len(uploaded_files)} uploaded PDFs")
            temp_dir = tempfile.mkdtemp()
            for uf in uploaded_files:
                temp_path = os.path.join(temp_dir, uf.name)
                with open(temp_path, "wb") as f:
                    f.write(uf.getbuffer())
                pdf_files.append(temp_path)
        
        if pdf_files:
            with st.spinner("⏳ Processing documents..."):
                try:
                    all_chunks = docling_chunks_from_pdfs(config, pdf_files)
                    
                    if not all_chunks:
                        st.error("❌ No chunks were extracted from the PDFs. Please check if the PDFs contain readable text.")
                    else:
                        st.write(f"✓ Created {len(all_chunks)} chunks")
                        
                        # Debug info
                        if all_chunks:
                            st.write(f"📊 Sample chunk preview:")
                            st.write(f"  - First chunk length: {len(all_chunks[0]['content'])} chars")
                            with st.expander("View first chunk sample"):
                                st.code(all_chunks[0]['content'][:500] + ("..." if len(all_chunks[0]['content']) > 500 else ""))
                        
                        rag_chain, retriever, vector_store, llm = rag_chain_from_params(config, all_chunks)
                        
                        st.session_state.rag_chain = rag_chain
                        st.session_state.retriever = retriever
                        st.session_state.vector_store = vector_store
                        st.session_state.llm = llm
                        st.session_state.documents_loaded = True
                        
                        st.success(f"✅ Loaded {len(pdf_files)} PDFs with {len(all_chunks)} chunks!")
                        
                        # Show storage location
                        if config.get("storage_path"):
                            st.info(f"💾 Database saved to: {config['storage_path']}")
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    logger.exception(f"Load error: {e}")
        else:
            st.error("❌ No PDFs found or uploaded.")
    
    if st.session_state.documents_loaded and st.session_state.vector_store:
        chunk_count = st.session_state.vector_store._collection.count()
        st.info(f"✅ System ready: {chunk_count} chunks loaded")
        
        # Display current settings
        with st.expander("🔧 Active Retrieval Settings"):
            st.write(f"**Search Type:** {config['search_type']}")
            st.write(f"**Top K:** {config['retrieval_k']}")
            st.write(f"**Reranking:** {'✅' if config['use_reranking'] else '❌'}")
            st.write(f"**Query Expansion:** {'✅' if config['use_query_expansion'] else '❌'}")
            if config.get("storage_path"):
                st.write(f"**Storage:** {config['storage_path']}")

# RIGHT COLUMN: Chat Interface
with col2:
    st.subheader("💬 Chat With Your Documents")
    
    if st.session_state.documents_loaded and st.session_state.rag_chain:
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 Sources"):
                        for idx, src in enumerate(msg["sources"], 1):
                            st.markdown(
                                f"**Source {idx}:** {src['metadata'].get('filename', src['metadata'].get('source', 'Unknown'))} "
                                f"(Page: {src['metadata'].get('page', 'N/A')})"
                            )
                            st.code(src["content"][:300] + ("..." if len(src["content"]) > 300 else ""))
        
        # Chat input
        user_input = st.chat_input("Ask a question about your documents...")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Process query
            query_to_use = user_input
            
            # Query expansion if enabled
            if config.get("use_query_expansion", False):
                with st.spinner("🔍 Expanding query..."):
                    try:
                        query_to_use = expand_query(user_input, st.session_state.llm)
                        logger.info(f"Expanded query: {query_to_use}")
                    except Exception as e:
                        logger.warning(f"Query expansion failed: {e}")
                        query_to_use = user_input
            
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get answer
                    answer = st.session_state.rag_chain.invoke(query_to_use)
                    
                    # Get sources (use original query for source display)
                    source_docs = st.session_state.retriever.get_relevant_documents(user_input)
                    sources = [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in source_docs
                    ]
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                
                except Exception as e:
                    st.error(f"❌ Error generating response: {e}")
                    logger.exception(f"Chat error: {e}")
            
            st.rerun()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    else:
        st.info(
            "👈 Please load PDFs using the panel on the left to start chatting with your documents."
        )
        
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. **Check Ollama connection** in the sidebar
        2. **Configure storage settings** (persistent recommended)
        3. **Upload PDFs** or specify a folder path
        4. **Click 'Load & Index PDFs'**
        5. **Start asking questions!**
        
        **Recommended Settings for Best Results:**
        - **Persistent Storage:** ✅ Enabled
        - **Search Type:** MMR (default)
        - **Top K:** 5 chunks
        - **Chunk Size:** 800 tokens
        - **Chunk Overlap:** 100 tokens
        
        **Your ChromaDB will be saved in:** `chromadb_storage` folder
        """)
