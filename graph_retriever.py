"""
Knowledge Graph Retriever for structured PDF extraction
"""

import spacy
import networkx as nx
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from utils import logger

class KnowledgeGraphRetriever:
    """Extract entities and relations, query via graph"""
    
    def __init__(self, chunks: List[Dict[str, Any]]):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.graph = nx.DiGraph()
        self.entity_to_chunks = {}
        self.chunks = chunks
        self._build_graph(chunks)
    
    def _build_graph(self, chunks: List[Dict[str, Any]]):
        """Build knowledge graph from chunks"""
        if not self.nlp:
            return
        
        logger.info(f"Building knowledge graph from {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            
            # Extract entities
            doc = self.nlp(content[:1000])  # Limit for performance
            
            for ent in doc.ents:
                entity_key = f"{ent.text}_{ent.label_}"
                
                # Add entity node
                if not self.graph.has_node(entity_key):
                    self.graph.add_node(entity_key, text=ent.text, type=ent.label_)
                
                # Track which chunks mention this entity
                if entity_key not in self.entity_to_chunks:
                    self.entity_to_chunks[entity_key] = []
                self.entity_to_chunks[entity_key].append(i)
                
                # Add edge from entity to document
                doc_id = metadata.get("filename", f"doc_{i}")
                self.graph.add_edge(entity_key, doc_id, relation="mentioned_in")
            
            # Extract co-occurrences (entities in same sentence)
            for sent in doc.sents:
                sent_ents = list(sent.ents)
                for j, ent1 in enumerate(sent_ents):
                    for ent2 in sent_ents[j+1:]:
                        key1 = f"{ent1.text}_{ent1.label_}"
                        key2 = f"{ent2.text}_{ent2.label_}"
                        self.graph.add_edge(key1, key2, relation="co_occurs")
        
        logger.info(f"Built knowledge graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def search_entities(self, query: str, top_k: int = 5) -> List[str]:
        """Find relevant entities from query"""
        if not self.nlp:
            return []
        
        query_doc = self.nlp(query)
        entity_scores = {}
        
        for ent in query_doc.ents:
            entity_key = f"{ent.text}_{ent.label_}"
            
            # Direct match
            if entity_key in self.graph.nodes:
                entity_scores[entity_key] = 10
            
            # Fuzzy match
            for node in self.graph.nodes:
                if ent.text.lower() in node.lower():
                    entity_scores[node] = entity_scores.get(node, 0) + 5
        
        # Sort by score
        sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
        return [ent for ent, _ in sorted_entities[:top_k]]
    
    def get_chunk_indices_for_entities(self, entities: List[str]) -> List[int]:
        """Get chunk indices that mention these entities"""
        chunk_indices = set()
        for entity in entities:
            if entity in self.entity_to_chunks:
                chunk_indices.update(self.entity_to_chunks[entity])
        return list(chunk_indices)
    
    def invoke(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents using knowledge graph"""
        if not self.nlp:
            return []
        
        # Find relevant entities
        entities = self.search_entities(query, top_k)
        
        if not entities:
            return []
        
        # Get chunks that mention these entities
        chunk_indices = self.get_chunk_indices_for_entities(entities)
        
        # Convert to Documents
        docs = []
        for idx in chunk_indices[:top_k]:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                doc = Document(
                    page_content=chunk["content"],
                    metadata={**chunk["metadata"], "graph_entities": entities}
                )
                docs.append(doc)
        
        return docs
