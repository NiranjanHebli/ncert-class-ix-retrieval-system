import re
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import tiktoken


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    content_type: str
    chapter: str
    chunk_id: str
    token_count: int
    semantic_boundary: str  # Type of boundary used: 'question', 'paragraph', 'example'


class ImprovedChunker:
    """
    Implements semantic-aware chunking strategies for different content types.
    """
    
    def __init__(
        self,
        max_tokens: int = 180,
    ):
        self.max_tokens = max_tokens
        self.tiktoken_enc = tiktoken.get_encoding('cl100k_base')
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken for accurate counting"""
        return len(self.tiktoken_enc.encode(text))
    
    def chunk_exercises(self, text: str, chapter: str) -> List[Chunk]:
        """
        Question-based chunking for exercises.
        Each complete question (including all sub-parts) becomes one chunk.
        """
        chunks = []
        
        # Split by question numbers (patterns like "1.", "2.", etc.)
        question_pattern = r'^\s*(\d+)\.\s*(.*?)(?=^\s*\d+\.|\Z)'
        questions = re.findall(question_pattern, text, re.MULTILINE | re.DOTALL)
        
        chunk_id = 0
        for q_num, q_text in questions:
            q_text = q_num + '. ' + q_text.strip()
            
            if self.count_tokens(q_text) > self.max_tokens:
                sub_chunks = self._split_long_question(q_text, chapter, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
            else:
                chunk = Chunk(
                    text=q_text,
                    content_type="exercise",
                    chapter=chapter,
                    chunk_id=f"{chapter}_exercise_{chunk_id}",
                    token_count=self.count_tokens(q_text),
                    semantic_boundary="question"
                )
                chunks.append(chunk)
                chunk_id += 1
                
        return chunks
    
    def _split_long_question(
        self,
        question_text: str,
        chapter: str,
        base_id: int,
    ) -> List[Chunk]:
        """Split a long question by sub-parts (a, b, c, etc.)"""
        chunks = []
        
        # Split by sub-parts
        subpart_pattern = r'([a-z])\)[\s]*([^a-z]*?)(?=[a-z]\)|\Z)'
        subparts = re.findall(subpart_pattern, question_text, re.DOTALL)
        
        for i, (letter, text) in enumerate(subparts):
            sub_text = f"{letter}) {text.strip()}"
            if self.count_tokens(sub_text) <= self.max_tokens:
                chunk = Chunk(
                    text=sub_text,
                    content_type="exercise",
                    chapter=chapter,
                    chunk_id=f"{chapter}_exercise_{base_id}_{i}",
                    token_count=self.count_tokens(sub_text),
                    semantic_boundary="question"
                )
                chunks.append(chunk)
            else:
                sentences = re.split(r'[.!?]+', sub_text)
                current_text = ""
                for sentence in sentences:
                    test_text = current_text + sentence + "."
                    if self.count_tokens(test_text) <= self.max_tokens:
                        current_text = test_text
                    else:
                        if current_text:
                            chunk = Chunk(
                                text=current_text.strip(),
                                content_type="exercise",
                                chapter=chapter,
                                chunk_id=f"{chapter}_exercise_{base_id}_{i}_{len(chunks)}",
                                token_count=self.count_tokens(current_text.strip()),
                                semantic_boundary="question"
                            )
                            chunks.append(chunk)
                        current_text = sentence + "."
                

                if current_text:
                    chunk = Chunk(
                        text=current_text.strip(),
                        content_type="exercise",
                        chapter=chapter,
                        chunk_id=f"{chapter}_exercise_{base_id}_{i}_{len(chunks)}",
                        token_count=self.count_tokens(current_text.strip()),
                        semantic_boundary="question"
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def chunk_concepts(self, text: str, chapter: str) -> List[Chunk]:
        """
        Semantic paragraph chunking for concepts.
        Natural paragraph breaks preserve conceptual flow.
        """
        chunks = []
        
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunk_id = 0
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            test_text = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if self.count_tokens(test_text) <= self.max_tokens:
                current_chunk = test_text
            else:
                if current_chunk:
                    chunk = Chunk(
                        text=current_chunk,
                        content_type="concept",
                        chapter=chapter,
                        chunk_id=f"{chapter}_concept_{chunk_id}",
                        token_count=self.count_tokens(current_chunk),
                        semantic_boundary="paragraph"
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                if self.count_tokens(paragraph) <= self.max_tokens:
                    current_chunk = paragraph
                else:
                    sentence_chunks = self._split_long_paragraph(paragraph, chapter, chunk_id)
                    chunks.extend(sentence_chunks)
                    chunk_id += len(sentence_chunks)
                    current_chunk = ""
        

        if current_chunk:
            chunk = Chunk(
                text=current_chunk,
                content_type="concept",
                chapter=chapter,
                chunk_id=f"{chapter}_concept_{chunk_id}",
                token_count=self.count_tokens(current_chunk),
                semantic_boundary="paragraph"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_long_paragraph(
        self,
        paragraph: str,
        chapter: str,
        base_id: int,
    ) -> List[Chunk]:
        """Split a long paragraph by sentences"""
        chunks = []
        sentences = re.split(r'[.!?]+', paragraph)
        
        current_text = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            test_text = current_text + ". " + sentence if current_text else sentence
            
            if self.count_tokens(test_text) <= self.max_tokens:
                current_text = test_text
            else:
                if current_text:
                    chunk = Chunk(
                        text=current_text + ".",
                        content_type="concept",
                        chapter=chapter,
                        chunk_id=f"{chapter}_concept_{base_id}_{len(chunks)}",
                        token_count=self.count_tokens(current_text + "."),
                        semantic_boundary="paragraph"
                    )
                    chunks.append(chunk)
                current_text = sentence
        

        if current_text:
            chunk = Chunk(
                text=current_text + ".",
                content_type="concept",
                chapter=chapter,
                chunk_id=f"{chapter}_concept_{base_id}_{len(chunks)}",
                token_count=self.count_tokens(current_text + "."),
                semantic_boundary="paragraph"
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_worked_examples(self, text: str, chapter: str) -> List[Chunk]:
        """
        Example-based chunking for worked examples.
        Each complete example (problem + full solution) becomes one chunk.
        """
        chunks = []
        
        # Split by example pattern (Example X.X)
        example_pattern = r'(Example\s+\d+\.\d+.*?)(?=Example\s+\d+\.\d+|\Z)'
        examples = re.findall(example_pattern, text, re.DOTALL)
        
        chunk_id = 0
        for example_text in examples:
            example_text = example_text.strip()
            
            if self.count_tokens(example_text) > self.max_tokens:
                sub_chunks = self._split_long_example(example_text, chapter, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
            else:
                chunk = Chunk(
                    text=example_text,
                    content_type="worked_example",
                    chapter=chapter,
                    chunk_id=f"{chapter}_example_{chunk_id}",
                    token_count=self.count_tokens(example_text),
                    semantic_boundary="example"
                )
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks
    
    def _split_long_example(
        self,
        example_text: str,
        chapter: str,
        base_id: int,
    ) -> List[Chunk]:
        """Split a long worked example by problem/solution sections"""
        chunks = []
        
        # Split by "Solution:" or similar patterns
        solution_pattern = r'(.*?)(?=##\s*Solution:|Solution:|##\s*Solution)'
        parts = re.split(solution_pattern, example_text, flags=re.IGNORECASE)
        
        if len(parts) > 1:
            problem = parts[1].strip()
            if problem and self.count_tokens(problem) <= self.max_tokens:
                chunk = Chunk(
                    text=problem,
                    content_type="worked_example",
                    chapter=chapter,
                    chunk_id=f"{chapter}_example_{base_id}_problem",
                    token_count=self.count_tokens(problem),
                    semantic_boundary="example"
                )
                chunks.append(chunk)
            
            solution = parts[2].strip() if len(parts) > 2 else ""
            if solution:
                if self.count_tokens(solution) <= self.max_tokens:
                    chunk = Chunk(
                        text=solution,
                        content_type="worked_example",
                        chapter=chapter,
                        chunk_id=f"{chapter}_example_{base_id}_solution",
                        token_count=self.count_tokens(solution),
                        semantic_boundary="example"
                    )
                    chunks.append(chunk)
                else:
                    solution_chunks = self._split_long_paragraph(solution, chapter, base_id + 100)
                    chunks.extend(solution_chunks)
        else:
            paragraph_chunks = self._split_long_paragraph(example_text, chapter, base_id)
            chunks.extend(paragraph_chunks)
        
        return chunks
    
    def detect_content_type(self, filename: str) -> str:
        """Detect content type from filename"""
        if 'exercise' in filename.lower():
            return 'exercise'
        elif 'concept' in filename.lower():
            return 'concept'
        elif 'example' in filename.lower():
            return 'worked_example'
        else:
            return 'concept'
    
    def chunk_file(self, file_path: str) -> List[Chunk]:
        """Chunk a single file using the appropriate strategy"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chapter = os.path.basename(file_path).replace('.txt', '')
        content_type = self.detect_content_type(file_path)
        
        if content_type == 'exercise':
            return self.chunk_exercises(text, chapter)
        elif content_type == 'concept':
            return self.chunk_concepts(text, chapter)
        elif content_type == 'worked_example':
            return self.chunk_worked_examples(text, chapter)
        else:
            return self.chunk_concepts(text, chapter)
    
    def chunk_directory(self, directory_path: str) -> List[Chunk]:
        """Chunk all files in a directory"""
        all_chunks = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    chunks = self.chunk_file(file_path)
                    all_chunks.extend(chunks)
        
        return all_chunks


class ImprovedVectorDatabase:
    """
    Enhanced vector database with improved chunking and BM25 retrieval.
    """
    
    def __init__(self):
        self.chunks = []
        self.bm25_index = None
        self.chunker = ImprovedChunker()
        
    def build_from_directory(self, directory_path: str):
        """Build database from directory using improved chunking"""
        self.chunks = self.chunker.chunk_directory(directory_path)
        
        self._build_bm25_index()
        
    def _build_bm25_index(self):
        """Build BM25 index from chunks"""
        from rank_bm25 import BM25Okapi
        import re
        
        tokenized_chunks = []
        for chunk in self.chunks:
            tokens = re.findall(r'\b\w+\b', chunk.text.lower())
            tokenized_chunks.append(tokens)
        
        self.bm25_index = BM25Okapi(tokenized_chunks)
        
    def retrieve_bm25(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve chunks using BM25"""
        if not self.bm25_index:
            raise ValueError("BM25 index not built. Call build_from_directory first.")
        
        import re
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        
        scores = self.bm25_index.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                'text': chunk.text,
                'chapter': chunk.chapter,
                'content_type': chunk.content_type,
                'semantic_boundary': chunk.semantic_boundary,
                'chunk_id': chunk.chunk_id,
                'token_count': chunk.token_count,
                'score': float(scores[idx])
            })
        
        return results
    
    def save_to_disk(self, db_path: str):
        """Save database to disk"""
        os.makedirs(db_path, exist_ok=True)
        
        chunks_data = []
        for chunk in self.chunks:
            chunks_data.append({
                'text': chunk.text,
                'content_type': chunk.content_type,
                'chapter': chunk.chapter,
                'chunk_id': chunk.chunk_id,
                'token_count': chunk.token_count,
                'semantic_boundary': chunk.semantic_boundary
            })
        
        with open(os.path.join(db_path, 'wk10_chunks.json'), 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    def load_from_disk(self, db_path: str):
        """Load database from disk"""
        chunks_file = os.path.join(db_path, 'wk10_chunks.json')
        
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"No saved database found at {db_path}")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        self.chunks = []
        for data in chunks_data:
            chunk = Chunk(
                text=data['text'],
                content_type=data['content_type'],
                chapter=data['chapter'],
                chunk_id=data['chunk_id'],
                token_count=data['token_count'],
                semantic_boundary=data['semantic_boundary']
            )
            self.chunks.append(chunk)
        
        self._build_bm25_index()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.chunks:
            return {}
        
        type_counts = {}
        boundary_counts = {}
        token_counts = []
        
        for chunk in self.chunks:
            type_counts[chunk.content_type] = type_counts.get(chunk.content_type, 0) + 1
            boundary_counts[chunk.semantic_boundary] = boundary_counts.get(chunk.semantic_boundary, 0) + 1
            token_counts.append(chunk.token_count)
        
        return {
            'total_chunks': len(self.chunks),
            'content_types': type_counts,
            'semantic_boundaries': boundary_counts,
            'avg_tokens': np.mean(token_counts),
            'max_tokens': max(token_counts),
            'min_tokens': min(token_counts),
            'chunks_over_limit': sum(1 for t in token_counts if t > 180)
        }
