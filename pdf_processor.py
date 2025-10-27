"""
PDF processing with ALL chunking methods - FIXED for current Docling API
"""

from typing import List, Dict, Any
from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from utils import logger

def docling_chunks_from_pdfs(config: Dict[str, Any], pdf_paths: List[str]) -> List[Dict[str, Any]]:
    """Extract and chunk text from PDFs using docling"""
    chunks = []
    
    try:
        converter = DocumentConverter()
        logger.debug(f"Initialized DocumentConverter")
        
        for pdf_path in pdf_paths:
            try:
                logger.debug(f"Processing PDF: {pdf_path}")
                
                doc_result = converter.convert(pdf_path)
                doc = doc_result.document
                
                text = ""
                if hasattr(doc, "export_to_markdown"):
                    text = doc.export_to_markdown()
                    logger.debug(f"Extracted using export_to_markdown")
                elif hasattr(doc, "text"):
                    text = doc.text
                    logger.debug(f"Extracted using text attribute")
                elif hasattr(doc, "main_text") and doc.main_text:
                    for item in doc.main_text:
                        if hasattr(item, "text") and item.text:
                            text += item.text + "\n"
                    logger.debug(f"Extracted using main_text")
                else:
                    logger.warning(f"Could not extract text from {pdf_path}")
                    continue
                
                metadata = {
                    "source": pdf_path,
                    "filename": pdf_path.split("/")[-1].split("\\")[-1]
                }
                
                if config.get("extract_tables", False):
                    try:
                        if hasattr(doc, "tables") and doc.tables:
                            for table in doc.tables:
                                if hasattr(table, "export_to_markdown"):
                                    table_text = table.export_to_markdown()
                                elif hasattr(table, "to_markdown"):
                                    table_text = table.to_markdown()
                                else:
                                    table_text = str(table)
                                text += f"\n[Table]\n{table_text}\n"
                                metadata["has_table"] = True
                            logger.debug(f"Extracted {len(doc.tables)} tables")
                    except Exception as e:
                        logger.warning(f"Table extraction failed: {e}")
                
                if not text.strip():
                    logger.error(f"No text extracted from {pdf_path}")
                    continue
                
                chunking_method = config.get("chunking_method", "docling")
                
                if chunking_method == "recursive":
                    text_chunks = recursive_chunking(text, config)
                elif chunking_method == "semantic":
                    text_chunks = semantic_chunking(text, config)
                elif chunking_method == "llm_powered":
                    text_chunks = llm_powered_chunking(text, config)
                elif chunking_method == "hybrid":
                    text_chunks = hybrid_multimodal_chunking(text, config)
                else:
                    text_chunks = _chunk_text(text, config.get("chunk_size", 512), config.get("chunk_overlap", 50))
                
                logger.debug(f"Created {len(text_chunks)} chunks using {chunking_method}")
                
                for i, chunk_text in enumerate(text_chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_id"] = f"{pdf_path}_{i}"
                    chunk_metadata["page"] = i + 1
                    chunks.append({
                        "content": chunk_text,
                        "metadata": chunk_metadata
                    })
                    
            except Exception as e:
                logger.error(f"Failed to process PDF {pdf_path}: {str(e)}")
                continue
        
        if not chunks:
            logger.error("No chunks extracted from any PDFs")
        else:
            logger.info(f"Extracted {len(chunks)} chunks from {len(pdf_paths)} PDFs")
        
        return chunks
        
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        return []

def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Simple text chunking with overlap"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            overlap_words = " ".join(current_chunk[-int(chunk_overlap/5):])
            current_chunk = overlap_words.split()
            current_length = len(overlap_words)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def recursive_chunking(text: str, config: Dict[str, Any]) -> List[str]:
    """Recursive character text splitting"""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get("chunk_size", 800),
            chunk_overlap=config.get("chunk_overlap", 100),
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        return splitter.split_text(text)
    except Exception as e:
        logger.error(f"Recursive chunking failed: {e}")
        return [text]

def semantic_chunking(text: str, config: Dict[str, Any]) -> List[str]:
    """Semantic chunking using embeddings"""
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        embeddings = OllamaEmbeddings(model=config.get("embedding_model", "nomic-embed-text"))
        splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        docs = splitter.create_documents([text])
        return [doc.page_content for doc in docs]
    except Exception as e:
        logger.error(f"Semantic chunking failed: {e}, falling back to recursive")
        return recursive_chunking(text, config)

def llm_powered_chunking(text: str, config: Dict[str, Any]) -> List[str]:
    """LLM-based proposition extraction"""
    try:
        llm = ChatOllama(
            model=config.get("llm_model", "gemma2:2b"),
            temperature=0,
            num_predict=500
        )
        
        prompt = f"""Break this text into atomic propositions (simple facts). Return one per line.

Text: {text[:2000]}

Propositions:"""
        
        response = llm.invoke(prompt)
        propositions = response.content.strip().split("\n")
        return [p.strip() for p in propositions if p.strip()]
    except Exception as e:
        logger.error(f"LLM chunking failed: {e}, falling back to recursive")
        return recursive_chunking(text, config)

def hybrid_multimodal_chunking(text: str, config: Dict[str, Any]) -> List[str]:
    """Hybrid chunking for multimodal content"""
    return recursive_chunking(text, config)
