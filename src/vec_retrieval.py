"""Vector database implementation with FAISS and BM25 retrieval."""

import numpy as np
import faiss
import re
import torch
import atexit
import os
from pathlib import Path
import pickle
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi


class VectorDatabase:
    """Vector database with chunking, embeddings, and retrieval."""
    
    def __init__(self, use_embeddings=True):
        """Initialize models and storage for vector database."""
        model_name = 'bert-base-uncased'
        self.bert_tok = AutoTokenizer.from_pretrained(model_name)
        
        self.use_embeddings = use_embeddings
        if self.use_embeddings:
            try:
                self.bert_mod = AutoModel.from_pretrained(model_name)
                self.bert_mod.eval()
                torch.set_num_threads(1)
            except Exception as e:
                print(f"Warning: Could not load BERT model, disabling embeddings: {e}")
                self.use_embeddings = False
        
        self.chunks = []
        self.bm25 = None
        self.faiss_idx = None
        self.embeds = None
        
        atexit.register(self.flush_vector_db)
        
    def chunk_text_bert(self, text, max_tokens=180, overlap=50):
        paragraphs = text.split('\n\n')
        all_chunks = []
        current_tokens = []
        
        for para in paragraphs:
            if not para.strip(): continue
            para_tokens = self.bert_tok.encode(para, add_special_tokens=False)
            
            if len(para_tokens) > max_tokens:
                for i in range(0, len(para_tokens), max_tokens - overlap):
                    all_chunks.append(self.bert_tok.decode(para_tokens[i:i + max_tokens]))
                continue

            if len(current_tokens) + len(para_tokens) > max_tokens:
                all_chunks.append(self.bert_tok.decode(current_tokens))
                overlap_tokens = current_tokens[-(overlap):] if len(current_tokens) > overlap else current_tokens
                current_tokens = overlap_tokens + para_tokens
            else:
                current_tokens.extend(para_tokens)
                
        if current_tokens:
            all_chunks.append(self.bert_tok.decode(current_tokens))
        return all_chunks
    
    def get_bert_embeddings(self, texts):
        if not self.use_embeddings: return None
        embeds = []
        batch_size = 4
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.bert_tok(batch, return_tensors='pt', truncation=True, max_length=512, padding=True)
                outputs = self.bert_mod(**inputs)
                batch_embeds = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeds.extend(batch_embeds)
        return np.array(embeds)
    
    def add_to_store(self, text: str, source_name="Unknown"):
        """Add new text to the existing chunk store without clearing it."""
        new_chunks_text = self.chunk_text_bert(text)
        
        start_id = len(self.chunks)
        for i, chunk_text in enumerate(new_chunks_text):
            self.chunks.append({
                'text': chunk_text,
                'chunk_id': start_id + i,
                'content_type': self._extract_content_type(chunk_text),
                'chapter': source_name,
                'section': self._extract_section(chunk_text)
            })
            
        # Re-build indexes after adding all chunks
        # In a production system, we'd do this once at the end of loading
        self._refresh_indexes()

    def _refresh_indexes(self):
        """Re-build BM25 and FAISS indexes from current chunks."""
        if not self.chunks:
            return
            
        # Build BM25 index
        tokenized_chunks = [chunk['text'].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        # Build FAISS index (Optional)
        if self.use_embeddings:
            try:
                embeddings = self.get_bert_embeddings([c['text'] for c in self.chunks])
                self.embeds = embeddings.astype(np.float32)
                self.faiss_idx = faiss.IndexFlatL2(self.embeds.shape[1])
                self.faiss_idx.add(self.embeds)
            except Exception as e:
                print(f"Warning: FAISS refresh failed: {e}")
                self.use_embeddings = False

    def build_chunk_store(self, text: str) -> list:
        """Compatibility method: clears store and builds from scratch."""
        self.chunks = []
        self.add_to_store(text)
        return self.chunks
    
    def build_chunk_store_from_file(self, file_path: str | Path) -> list:
        """Add a single file to the store."""
        source_name = Path(file_path).stem
        with open(file_path, 'r', encoding='utf-8') as f:
            self.add_to_store(f.read(), source_name=source_name)
        return self.chunks
    
    def _extract_content_type(self, text):
        t = text.lower()
        if 'example' in t: return "example"
        if 'solution' in t: return "solution"
        if 'exercise' in t: return "exercise"
        return "content"
    
    def _extract_chapter(self, text):
        m = re.search(r'(?:Chapter|iesc1)(\d+)', text, re.I)
        return f"Chapter {m.group(1)}" if m else "Unknown"
    
    def _extract_section(self, text):
        m = re.search(r'## \*\*(\d+\.\d+)\*\*', text)
        return m.group(1) if m else "Unknown"
    
    def retrieve_faiss(self, query, k=3):
        if not self.use_embeddings or self.faiss_idx is None: return []
        query_embed = self.get_bert_embeddings([query]).astype(np.float32)
        dists, idxs = self.faiss_idx.search(query_embed, k)
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            if idx != -1 and idx < len(self.chunks):
                c = self.chunks[idx].copy()
                c["similarity_score"] = float(1 / (1 + dist))
                results.append(c)
        return results
    
    def retrieve_bm25(self, query, k=3):
        if self.bm25 is None: return []
        scores = self.bm25.get_scores(query.lower().split())
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top_k]

    def retrieve_bm25_with_scores(self, query, k=3):
        if self.bm25 is None: return []
        scores = self.bm25.get_scores(query.lower().split())
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for i in top_k:
            chunk = self.chunks[i].copy()
            chunk['bm25_score'] = scores[i]
            results.append(chunk)
        return results
    
    def save_to_disk(self, directory: str | Path):
        """Save the database and index to a directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
            
        # Save BM25
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
            
        # Save FAISS index if it exists
        if self.faiss_idx is not None:
            faiss.write_index(self.faiss_idx, str(path / "faiss.idx"))
            np.save(path / "embeds.npy", self.embeds)
            
        print(f"Database saved to {directory}")

    def load_from_disk(self, directory: str | Path):
        """Load the database and index from a directory."""
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Database directory {directory} not found.")
            
        # Load chunks
        with open(path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
            
        # Load BM25
        bm25_path = path / "bm25.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            # Re-build BM25 if missing
            self._refresh_indexes()
            
        # Load FAISS index if it exists
        faiss_path = path / "faiss.idx"
        if faiss_path.exists():
            self.faiss_idx = faiss.read_index(str(faiss_path))
            self.embeds = np.load(path / "embeds.npy")
            self.use_embeddings = True
            
        print(f"Database loaded from {directory} ({len(self.chunks)} chunks)")

    def flush_vector_db(self):
        self.faiss_idx = None
        self.embeds = None
        self.bm25 = None
        self.chunks = []
        self.bert_mod = None
        self.bert_tok = None