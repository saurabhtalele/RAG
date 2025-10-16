import tempfile
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker, HierarchicalChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from unstructured.partition.pdf import partition_pdf
import json
from utils import sanitize_metadata, logger, redact_pii

def docling_chunks_from_pdfs(config: Dict[str, Any], pdf_paths: List[str]) -> List[Dict[str, Any]]:
    st.write("📋 Initializing PDF converter...")
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = config["ocr_enabled"]
        pipeline_options.do_table_structure = config["extract_tables"]
        doc_converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            pipeline_options=pipeline_options
        )
    except TypeError:
        st.write("  ⚠️ Using legacy Docling API")
        doc_converter = DocumentConverter(allowed_formats=[InputFormat.PDF])
    
    all_chunks = []
    total_pdfs = len(pdf_paths)
    for idx, pdf_path in enumerate(pdf_paths, 1):
        try:
            filename = Path(pdf_path).name
            st.write(f"📄 Processing {idx}/{total_pdfs}: {filename}...")
            result = doc_converter.convert(pdf_path)
            doc = result.document
            markdown_text = doc.export_to_markdown()
            
            if config["chunking_method"] == "recursive":
                chunks = recursive_chunking(markdown_text, config)
            elif config["chunking_method"] == "semantic":
                chunks = semantic_chunking(markdown_text, config)
            elif config["chunking_method"] == "llm_powered":
                chunks = llm_powered_chunking(doc, config, st.session_state.get("llm"))
            elif config["chunking_method"] == "hybrid":
                chunks = hybrid_multimodal_chunking(pdf_path, config)
            else:
                chunker_class = HierarchicalChunker if config["use_hierarchical_retrieval"] else HybridChunker
                chunker = chunker_class(max_tokens=config["chunk_size"], overlap=config["chunk_overlap"])
                chunk_iter = chunker.chunk(doc)
                chunks = []
                for chunk in chunk_iter:
                    chunk_text = getattr(chunk, 'text', getattr(chunk, 'content', ''))
                    if not (chunk_text and chunk_text.strip()):
                        continue
                    metadata = {
                        "source": str(pdf_path),
                        "filename": filename,
                        "page": getattr(chunk, "page_num", None),
                        "chunk_type": config["chunking_method"]
                    }
                    if hasattr(chunk, 'meta') and chunk.meta:
                        try:
                            meta_dict = chunk.meta.to_dict() if hasattr(chunk.meta, 'to_dict') else dict(chunk.meta)
                            if isinstance(meta_dict, dict):
                                metadata.update(meta_dict)
                        except Exception as e:
                            logger.warning(f"Metadata issue: {e}")
                    chunks.append({"content": chunk_text, "metadata": metadata})
            
            for chunk in chunks:
                content = chunk["content"]
                if config["enable_pii_redaction"]:
                    content = redact_pii(content)
                metadata = sanitize_metadata({
                    **chunk.get("metadata", {}),
                    "source": str(pdf_path),
                    "filename": filename,
                    "chunk_type": config["chunking_method"]
                })
                all_chunks.append({"content": content, "metadata": metadata})
            st.write(f"  └─ ✓ Extracted {len(chunks)} {config['chunking_method']} chunks")
        except Exception as e:
            st.warning(f"⚠️ Failed to process {Path(pdf_path).name}: {e}")
            logger.exception(f"PDF error: {e}")
    return all_chunks

def recursive_chunking(text: str, config: Dict) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    splits = splitter.split_text(text)
    return [{"content": s, "metadata": {}} for s in splits]

def semantic_chunking(text: str, config: Dict) -> List[Dict]:
    embeddings = OllamaEmbeddings(model=config["embedding_model"])
    splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    splits = splitter.split_text(text)
    return [{"content": s, "metadata": {"chunk_type": "semantic"}} for s in splits]

def llm_powered_chunking(doc, config: Dict, llm) -> List[Dict]:
    if not llm:
        return recursive_chunking(doc.export_to_markdown(), config)
    markdown_text = doc.export_to_markdown()
    pre_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    sections = pre_splitter.split_text(markdown_text)
    chunks = []
    for section in sections:
        prompt = f"""
Break the following text into concise, standalone factual statements (propositions).
Each should be self-contained and no longer than {config['chunk_size']} characters.
Return as a numbered list:
Text: {section}
Propositions:
"""
        try:
            response = llm.invoke(prompt)
            propositions = response.content if hasattr(response, 'content') else str(response)
            for line in propositions.split('\n'):
                if line.strip() and line[0].isdigit() and '.' in line:
                    prop = line[line.find('.')+1:].strip()
                    if prop:
                        chunks.append({"content": prop, "metadata": {"chunk_type": "proposition"}})
        except Exception as e:
            logger.warning(f"LLM chunking failed: {e}")
            chunks.extend(recursive_chunking(section, config))
    return chunks if chunks else recursive_chunking(markdown_text, config)

def hybrid_multimodal_chunking(pdf_path: str, config: Dict) -> List[Dict]:
    if config["multimodal_enabled"]:
        try:
            elements = partition_pdf(pdf_path, strategy="hi_res")
            chunks = []
            for elem in elements:
                if elem.category == "Table":
                    content = json.dumps(elem.metadata.text_as_html)
                elif elem.category == "Image":
                    content = f"[Image placeholder: {elem.metadata.image_path}]"
                else:
                    content = elem.text
                if content and content.strip():
                    chunks.append({"content": content, "metadata": {"category": elem.category}})
            return chunks
        except Exception as e:
            st.warning(f"⚠️ Hybrid chunking failed: {e}")
    return recursive_chunking(Path(pdf_path).read_text(), config)


# ===== PDF PROCESSOR CLASS =====

class PDFProcessor:
    """
    PDF Processor class that wraps the chunking functions
    and provides vectorstore creation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PDF processor with configuration"""
        self.config = config
        self.embeddings = None
    
    def _get_embeddings(self):
        """Lazy initialization of embeddings"""
        if self.embeddings is None:
            self.embeddings = OllamaEmbeddings(
                model=self.config.get("embedding_model", "nomic-embed-text:latest")
            )
        return self.embeddings
    
    def process_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """
        Process PDF files and return LangChain Document objects
        
        Args:
            pdf_paths: List of paths to PDF files
        
        Returns:
            List of LangChain Document objects
        """
        # Use existing function to get chunks
        chunks = docling_chunks_from_pdfs(self.config, pdf_paths)
        
        # Convert to LangChain Document objects
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["content"],
                metadata=chunk.get("metadata", {})
            )
            documents.append(doc)
        
        return documents
    
    def create_vectorstore(self, documents: List[Document]):
        """
        Create Chroma vectorstore from documents
        
        Args:
            documents: List of LangChain Document objects
        
        Returns:
            Chroma vectorstore instance
        """
        embeddings = self._get_embeddings()
        
        # Create vectorstore
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="rag_collection"
        )
        
        return vectorstore