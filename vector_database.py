"""Vector database implementation with FAISS and BM25 retrieval."""

import numpy as np
import faiss
import re
import torch
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi


class VectorDatabase:
    """Vector database with chunking, embeddings, and retrieval."""
    
    def __init__(self):
        """Initialize models and storage."""
        self.bert_tok = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_mod = AutoModel.from_pretrained('bert-base-uncased')
        self.t5_tok = AutoTokenizer.from_pretrained('t5-small')
        self.t5_mod = AutoModel.from_pretrained('t5-small')
        
        self.chunks = []
        self.bm25 = None
        self.faiss_idx = None
        self.embeds = None
        
    def chunk_text_bert(self, text, max_tokens=500, overlap=50):
        """Chunk text using BERT tokenizer."""
        chaps = re.split(r'Chapter \d+:', text)
        chunks = []
        
        for chap in chaps[1:]:
            chap_match = re.search(r'Chapter (\d+):', text)
            if chap_match:
                chap_hdr = f"Chapter {chap_match.group(1)}:"
                chap = chap_hdr + chap
            
            tokens = self.bert_tok.encode(chap, add_special_tokens=False)
            
            for i in range(0, len(tokens), max_tokens - overlap):
                chunk_tokens = tokens[i:i + max_tokens]
                chunk_text = self.bert_tok.decode(chunk_tokens)
                chunks.append(chunk_text)
                
                if i + max_tokens >= len(tokens):
                    break
                
        return chunks
    
    def chunk_text_t5(self, text, max_tokens=300, overlap=50):
        """Chunk text using T5 tokenizer."""
        tokens = self.t5_tok.encode(text, add_special_tokens=False)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.t5_tok.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if i + max_tokens >= len(tokens):
                break
                
        return chunks
    
    def get_bert_embeddings(self, texts):
        """Get BERT embeddings for texts."""
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
    
    def get_t5_embeddings(self, texts):
        """Get T5 embeddings for texts."""
        embeds = []
        
        with torch.no_grad():
            for txt in texts:
                inputs = self.t5_tok(txt, return_tensors='pt', 
                                   truncation=True, max_length=512, 
                                   padding=True)
                outputs = self.t5_mod(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeds.append(embedding)
                
        return np.array(embeds)
    
    def classify_chunk(self, text):
        """Classify content based on keywords."""
        txt = text.lower()
        
        if any(w in txt for w in ['reaction', 'equation', 'chemical', 'atom', 'bond', 'reactant', 'product', 'combination', 'decomposition', 'displacement']):
            return 'chemical_reactions'
        elif any(w in txt for w in ['acid', 'base', 'ph', 'sour', 'bitter', 'salt', 'litmus', 'dissociate', 'neutralization']):
            return 'acids_bases'
        elif any(w in txt for w in ['metal', 'non-metal', 'conductor', 'reactivity', 'malleable', 'ductile', 'shiny', 'brittle']):
            return 'metals_nonmetals'
        elif any(w in txt for w in ['carbon', 'organic', 'hydrocarbon', 'functional', 'alcohol', 'aldehyde', 'ketone', 'isomer']):
            return 'carbon_compounds'
        else:
            return 'general'
    
    def build_chunk_store(self, text):
        """Build chunk store with metadata using FAISS."""
        bert_chunks = self.chunk_text_bert(text, max_tokens=500, overlap=50)
        t5_chunks = self.chunk_text_t5(text, max_tokens=300, overlap=50)
        
        print(f"BERT chunks created: {len(bert_chunks)}")
        print(f"T5 chunks created: {len(t5_chunks)}")
        
        print("\n=== BERT CHUNKS (first 5) ===")
        for i, chunk in enumerate(bert_chunks[:5]):
            tokens = self.bert_tok.encode(chunk)
            print(f"Chunk {i+1} ({len(tokens)} tokens): {chunk[:100]}...")
            
        print("\n=== T5 CHUNKS (first 5) ===")
        for i, chunk in enumerate(t5_chunks[:5]):
            tokens = self.t5_tok.encode(chunk)
            print(f"Chunk {i+1} ({len(tokens)} tokens): {chunk[:100]}...")
        
        raw_chunks = bert_chunks
        
        self.chunks = []
        for i, txt in enumerate(raw_chunks):
            self.chunks.append({
                "id": i,
                "text": txt,
                "chapter": self._extract_chapter(txt),
                "section": self._extract_section(txt),
                "content_type": self.classify_chunk(txt)
            })
        
        self.embeds = self.get_bert_embeddings([c["text"] for c in self.chunks])
        
        dim = self.embeds.shape[1]
        self.faiss_idx = faiss.IndexFlatL2(dim)
        self.faiss_idx.add(self.embeds.astype(np.float32))
        
        corpus = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)
        
        print(f"\nBuilt chunk store with {len(self.chunks)} chunks")
        print(f"FAISS index dimension: {dim}")
        print(f"BM25 corpus size: {len(corpus)}")
        
        return self.chunks
    
    def _extract_chapter(self, text):
        """Extract chapter information from text."""
        chap_match = re.search(r'Chapter (\d+):', text)
        if chap_match:
            return f"Chapter {chap_match.group(1)}"
        return "Unknown"
    
    def _extract_section(self, text):
        """Extract section information from text."""
        sec_match = re.search(r'(\d+\.\d+)', text)
        if sec_match:
            return sec_match.group(1)
        return "Unknown"
    
    def retrieve_faiss(self, query, k=3):
        """Retrieve chunks using FAISS similarity search."""
        query_embed = self.get_bert_embeddings([query])
        query_embed = query_embed.astype(np.float32)
        
        dists, idxs = self.faiss_idx.search(query_embed, k)
        
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk["similarity_score"] = float(1 / (1 + dist))
                results.append(chunk)
                
        return results
    
    def retrieve_bm25(self, query, k=3):
        """Retrieve chunks using BM25."""
        scores = self.bm25.get_scores(query.lower().split())
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_k:
            chunk = self.chunks[idx].copy()
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)
            
        return results
    
    def test_retrieval(self):
        """Test retrieval with sample questions."""
        questions = [
            "What are the different types of chemical reactions?",
            "How do acids and bases differ in properties?",
            "What makes metals good conductors of electricity?"
        ]
        
        print("\n=== BM25 RETRIEVAL TEST ===")
        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i}: {q}")
            results = self.retrieve_bm25(q, k=3)
            
            for j, res in enumerate(results, 1):
                print(f"  Result {j} (Score: {res['bm25_score']:.3f}):")
                print(f"    Chapter: {res['chapter']}")
                print(f"    Content Type: {res['content_type']}")
                print(f"    Text: {res['text'][:150]}...")
                print()

if __name__ == "__main__":
    with open('sample_science_text.txt', 'r') as f:
        txt = f.read()
    
    db = VectorDatabase()
    chunks = db.build_chunk_store(txt)
    
    db.test_retrieval()
