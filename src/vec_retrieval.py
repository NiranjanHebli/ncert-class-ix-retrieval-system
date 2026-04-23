"""Vector database implementation with FAISS and BM25 retrieval."""

import numpy as np
import faiss
import re
import torch
import atexit
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi


class VectorDatabase:
    """Vector database with chunking, embeddings, and retrieval."""
    
    def __init__(self):
        """Initialize models and storage for vector database."""
        self.bert_tok = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_mod = AutoModel.from_pretrained('bert-base-uncased')
        
        self.chunks = []
        self.bm25 = None
        self.faiss_idx = None
        self.embeds = None
        
        # Register cleanup on program exit
        atexit.register(self.flush_vector_db)
        
    def chunk_text_bert(self, text, max_tokens=500, overlap=50):
        """Chunk text using BERT tokenizer with IESC102 section awareness.
        
        Args:
            text: Input text to chunk
            max_tokens: Maximum tokens per chunk
            overlap: Token overlap between chunks
            
        Returns:
            list: Text chunks preserving section boundaries
        """
        sections = re.split(r'## \*\*(\d+\.\d+)\*\*|## \*\*([^\d]+)\*\*', text)
        chunks = []
        
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                section_num = sections[i]
                section_title = sections[i + 1]
                section_content = f"## **{section_num}** {section_title}"
                
                activities = re.split(r'## _\*\*Activity ______________ (\d+\.\d+)\*\*_', section_content)
                
                if len(activities) > 1:
                    for j in range(1, len(activities), 2):
                        if j + 1 < len(activities):
                            activity_num = activities[j]
                            activity_content = activities[j + 1]
                            full_activity = f"## _**Activity ______________ {activity_num}**_ {activity_content}"
                            
                            tokens = self.bert_tok.encode(full_activity, add_special_tokens=False)
                            
                            for k in range(0, len(tokens), max_tokens - overlap):
                                chunk_tokens = tokens[k:k + max_tokens]
                                chunk_text = self.bert_tok.decode(chunk_tokens)
                                chunks.append(chunk_text)
                                
                                if k + max_tokens >= len(tokens):
                                    break
                else:
                    tokens = self.bert_tok.encode(section_content, add_special_tokens=False)
                    
                    for k in range(0, len(tokens), max_tokens - overlap):
                        chunk_tokens = tokens[k:k + max_tokens]
                        chunk_text = self.bert_tok.decode(chunk_tokens)
                        chunks.append(chunk_text)
                        
                        if k + max_tokens >= len(tokens):
                            break
                            
        return chunks
    
    
    def get_bert_embeddings(self, texts):
        """Get BERT embeddings for texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy.ndarray: BERT embeddings for input texts
        """
        embeds = []
        
        with torch.no_grad():
            for txt in texts:
                inputs = self.bert_tok(txt, return_tensors='pt', 
                                     truncation=True, max_length=512, 
                                     padding=True)
                outputs = self.bert_mod(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeds.append(embedding)
                
        return np.array(embeds)
    
    
    def build_chunk_store(self, text: str) -> list:
        """Build chunk store from text with metadata using FAISS and BM25.
        
        Args:
            text: Input text to process
            
        Returns:
            list: Processed chunks with metadata
        """
        chunks = self.chunk_text_bert(text)
        
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = {
                'text': chunk_text,
                'chunk_id': i,
                'content_type': self._extract_content_type(chunk_text),
                'chapter': self._extract_chapter(chunk_text),
                'section': self._extract_section(chunk_text)
            }
            processed_chunks.append(chunk)
        
        self.chunks = processed_chunks
        
        # Build BM25 index
        tokenized_chunks = [chunk['text'].split() for chunk in processed_chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        # Build FAISS index
        chunk_texts = [chunk['text'] for chunk in processed_chunks]
        self.embeds = self.get_bert_embeddings(chunk_texts)
        self.embeds = self.embeds.astype(np.float32)
        
        dimension = self.embeds.shape[1]
        self.faiss_idx = faiss.IndexFlatL2(dimension)
        self.faiss_idx.add(self.embeds)
        
        return processed_chunks
    
    def build_chunk_store_from_file(self, file_path: str | Path) -> list:
        """Build chunk store from text file with metadata using FAISS.
        
        Args:
            file_path: Path to the input document file
            
        Returns:
            list: Processed chunks with metadata
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.build_chunk_store(text)
    
    def _extract_content_type(self, text):
        """Extract content type from IESC102 text.
        
        Args:
            text: Text chunk to analyze
            
        Returns:
            str: Content type (definition, example, activity, etc.) or "content"
        """
        if 'definition' in text.lower():
            return "definition"
        elif 'example' in text.lower():
            return "example"
        elif 'activity' in text.lower():
            return "activity"
        elif 'exercise' in text.lower():
            return "exercise"
        else:
            return "content"
    
    def _extract_chapter(self, text):
        """Extract chapter information from IESC102 text.
        
        Args:
            text: Text chunk to analyze
            
        Returns:
            str: Chapter identifier or "Unknown"
        """
        chap_match = re.search(r'## Chapter \*\*(\d+)\*\*', text)
        if chap_match:
            return f"Chapter {chap_match.group(1)}"
        return "Unknown"
    
    def _extract_section(self, text):
        """Extract section information from IESC102 text.
        
        Args:
            text: Text chunk to analyze
            
        Returns:
            str: Section identifier or "Unknown"
        """
        sec_match = re.search(r'## \*\*(\d+\.\d+)\*\*', text)
        if sec_match:
            return sec_match.group(1)
        
        act_match = re.search(r'Activity ______________ (\d+\.\d+)', text)
        if act_match:
            return f"Activity {act_match.group(1)}"
            
        return "Unknown"
    
    def retrieve_faiss(self, query, k=3):
        """Retrieve chunks using FAISS similarity search.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            list: Top-k most similar chunks with similarity scores
        """
        query_embed = self.get_bert_embeddings([query])
        query_embed = query_embed.astype(np.float32)
        
        dists, idxs = self.faiss_idx.search(query_embed, k)
        
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk["similarity_score"] = float(np.exp(-dist))
                results.append(chunk)
                
        return results
    
    def retrieve_bm25(self, query, k=3):
        """Retrieve chunks using BM25.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            list: Top-k most relevant chunks with BM25 scores
        """
        scores = self.bm25.get_scores(query.lower().split())
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_k:
            chunk = self.chunks[idx].copy()
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)
            
        return results
    
    def flush_vector_db(self):
        """Flush and clean up vector database resources on program exit.
        
        Clears FAISS index, embeddings, BM25 index, chunks, and model references
        to free memory and prevent resource leaks.
        """
        try:
            if self.faiss_idx is not None:
                self.faiss_idx.reset()
                self.faiss_idx = None
            
            if self.embeds is not None:
                self.embeds = None
            
            self.bm25 = None
            self.chunks = []
            self.bert_mod = None
            self.bert_tok = None
            
            print("Vector database flushed successfully")
            
        except Exception as e:
            print(f"Error flushing vector database: {e}")