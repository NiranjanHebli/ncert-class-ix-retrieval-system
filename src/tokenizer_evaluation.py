import json
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import pandas as pd

DEFAULT_TOKENIZERS = {
    'gpt2': 'gpt2',
    'bert-base-uncased': 'bert-base-uncased',
    't5-small': 't5-small'
}

DEFAULT_CHUNK_SIZES = [100, 200, 300, 500, 800, 1000]
DEFAULT_OVERLAPS = [0, 50, 100, 200]

SCORING_WEIGHTS = {
    'compression_ratio': 0.33,
    'token_consistency': 0.33,
    'encoding_speed': 0.33
}

OUTPUT_FILES = {
    'results': '../data/tokenizer_results.json',
    'comparison': '../data/tokenizer_comparison.csv'
}

class TokenizerEvaluator:
    """Evaluates different tokenizers with various chunking strategies."""
    
    def __init__(self, tokenizer_configs: Optional[Dict[str, str]] = None):
        """Initialize the evaluator with specified tokenizers.
        
        Args:
            tokenizer_configs: Dictionary mapping tokenizer names to model names.
                              If None, uses default tokenizers.
        """
        configs = tokenizer_configs or DEFAULT_TOKENIZERS
        self.tokenizers = self._load_tokenizers(configs)
    
    def _load_tokenizers(self, configs: Dict[str, str]) -> Dict[str, AutoTokenizer]:
        """Load and configure tokenizers."""
        tks = {}
        for name, model in configs.items():
            try:
                tk = AutoTokenizer.from_pretrained(model)
                if name == 'gpt2' and tk.pad_token is None:
                    tk.pad_token = tk.eos_token
                tks[name] = tk
            except Exception as e:
                print(f"Warning: Failed to load tokenizer {name}: {e}")
        return tks
    
    def load_sample_text(self, filepath: str) -> str:
        """Load sample text from file.
        
        Args:
            filepath: Path to the text file
            
        Returns:
            The loaded text content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If there's an error reading the file
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {filepath}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except IOError as e:
            raise IOError(f"Error reading file {filepath}: {e}")
    
    def chunk_text(self, text: str, size: int, overlap: int) -> List[str]:
        """Split text into chunks with specified size and overlap.
        
        Args:
            text: The text to chunk
            size: Size of each chunk in characters
            overlap: Overlap between consecutive chunks
            
        Returns:
            List of text chunks
            
        Raises:
            ValueError: If overlap >= size or parameters are invalid
        """
        if overlap >= size:
            raise ValueError(f"Overlap ({overlap}) must be less than chunk size ({size})")
        if size <= 0 or overlap < 0:
            raise ValueError("Chunk size must be positive and overlap must be non-negative")
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + size, len(text))
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start = end - overlap
        
        return chunks
    
    def evaluate_tokenizer(self, name: str, text: str, sizes: List[int], 
                          overlaps: List[int]) -> Dict[str, Dict]:
        """Evaluate a tokenizer with different chunking parameters.
        
        Args:
            name: Name of the tokenizer to evaluate
            text: Text to evaluate on
            sizes: List of chunk sizes to test
            overlaps: List of overlaps to test
            
        Returns:
            Dictionary mapping configuration names to metrics
            
        Raises:
            KeyError: If tokenizer name is not found
        """
        if name not in self.tokenizers:
            raise KeyError(f"Tokenizer '{name}' not found. Available: {list(self.tokenizers.keys())}")
        
        tk = self.tokenizers[name]
        results = {}
        
        for size in sizes:
            for overlap in overlaps:
                try:
                    metrics = self._evaluate_configuration(tk, text, size, overlap)
                    key = f"{size}_{overlap}"
                    results[key] = metrics
                except ValueError as e:
                    print(f"Skipping invalid configuration {size}_{overlap}: {e}")
                    continue
        
        return results
    
    def _evaluate_configuration(self, tk: AutoTokenizer, text: str, 
                               size: int, overlap: int) -> Dict:
        """Evaluate a single chunking configuration."""
        chunks = self.chunk_text(text, size, overlap)
        
        metrics = self._compute_chunk_metrics(tk, chunks)
        vocab_util = self._compute_vocab_utilization(tk, chunks)
        compression_ratio = len(text) / metrics['total_tokens']
        
        return {
            'chunk_size': size,
            'overlap': overlap,
            'num_chunks': len(chunks),
            'total_tokens': metrics['total_tokens'],
            'avg_tokens_per_chunk': metrics['avg_tokens'],
            'std_tokens_per_chunk': metrics['std_tokens'],
            'avg_encoding_time': metrics['avg_encoding_time'],
            'total_encoding_time': metrics['total_encoding_time'],
            'vocab_utilization': vocab_util,
            'compression_ratio': compression_ratio,
            'chunk_token_counts': metrics['token_counts']
        }
    
    def _compute_chunk_metrics(self, tk: AutoTokenizer, chunks: List[str]) -> Dict:
        """Compute tokenization metrics for chunks."""
        counts = []
        times = []
        total = 0
        
        for chunk in chunks:
            start = time.time()
            tokens = tk.encode(chunk)
            end = time.time()
            
            count = len(tokens)
            enc_time = end - start
            
            counts.append(count)
            times.append(enc_time)
            total += count
        
        return {
            'total_tokens': total,
            'avg_tokens': np.mean(counts),
            'std_tokens': np.std(counts),
            'avg_encoding_time': np.mean(times),
            'total_encoding_time': sum(times),
            'token_counts': counts
        }
    
    def _compute_vocab_utilization(self, tk: AutoTokenizer, chunks: List[str]) -> float:
        """Compute vocabulary utilization ratio."""
        all_tokens = []
        for chunk in chunks:
            all_tokens.extend(tk.encode(chunk))
        
        unique = len(set(all_tokens))
        return unique / tk.vocab_size
    
    def run_evaluation(self, text_file: str, sizes: List[int] = None, 
                       overlaps: List[int] = None) -> Dict[str, Dict]:
        """Run complete evaluation for all tokenizers.
        
        Args:
            text_file: Path to the text file
            sizes: List of chunk sizes to test (uses default if None)
            overlaps: List of overlaps to test (uses default if None)
            
        Returns:
            Dictionary mapping tokenizer names to their results
        """
        sizes = sizes or DEFAULT_CHUNK_SIZES
        overlaps = overlaps or DEFAULT_OVERLAPS
        
        text = self.load_sample_text(text_file)
        print(f"Loaded text with {len(text)} characters")
        
        results = {}
        
        for name in self.tokenizers.keys():
            print(f"Evaluating {name}...")
            try:
                res = self.evaluate_tokenizer(name, text, sizes, overlaps)
                results[name] = res
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                continue
        
        return results
    
    def analyze_results(self, results: Dict[str, Dict]) -> Dict[str, Dict]:
        """Analyze results and determine best parameters.
        
        Args:
            results: Results dictionary from run_evaluation
            
        Returns:
            Analysis dictionary with best configurations for each tokenizer
        """
        analysis = {}
        
        for name, tk_results in results.items():
            if not tk_results:
                print(f"No results for tokenizer {name}")
                continue
                
            best_config, best_score = self._find_best_configuration(tk_results)
            
            analysis[name] = {
                'best_config': best_config,
                'best_score': best_score,
                'best_metrics': tk_results[best_config] if best_config else None
            }
        
        return analysis
    
    def _find_best_configuration(self, results: Dict[str, Dict]) -> Tuple[Optional[str], float]:
        """Find the best configuration for a tokenizer."""
        best_config = None
        best_score = -float('inf')
        
        for config_name, metrics in results.items():
            score = self._compute_configuration_score(metrics)
            
            if score > best_score:
                best_score = score
                best_config = config_name
        
        return best_config, best_score
    
    def _compute_configuration_score(self, metrics: Dict) -> float:
        """Compute a composite score for a configuration."""
        w = SCORING_WEIGHTS
        
        # Context retention bonus: reward configurations with meaningful overlap
        overlap_ratio = metrics['overlap'] / metrics['chunk_size']
        # Optimal overlap is between 10-30% of chunk size for context retention
        if 0.1 <= overlap_ratio <= 0.3:
            context_bonus = 1.2  # 20% bonus for optimal overlap
        elif overlap_ratio > 0:
            context_bonus = 1.1  # 10% bonus for any positive overlap
        else:
            context_bonus = 0.8  # 20% penalty for no overlap
        
        effective_compression = metrics['compression_ratio']
        comp_score = effective_compression * context_bonus
        
        cons_score = 1 / (metrics['std_tokens_per_chunk'] + 1)
        
        speed_score = 1 / (metrics['avg_encoding_time'] * 1000 + 1)
        
        comp_normalized = min(comp_score / 5.0, 1.0)  # Normalize compression to 0-1 (max ~5)
        cons_normalized = min(cons_score, 1.0)  # Already 0-1 range
        speed_normalized = min(speed_score, 1.0)  # Already 0-1 range
        
        return (comp_normalized * w['compression_ratio'] + 
                cons_normalized * w['token_consistency'] + 
                speed_normalized * w['encoding_speed'])
    
    def save_results(self, results: Dict[str, Dict], analysis: Dict[str, Dict], 
                     filename: str = None) -> None:
        """Save results to JSON file.
        
        Args:
            results: Results dictionary from run_evaluation
            analysis: Analysis dictionary from analyze_results
            filename: Output filename (uses default if None)
        """
        filename = filename or OUTPUT_FILES['results']
        
        output = {
            'results': results,
            'analysis': analysis,
            'summary': self._generate_summary(results, analysis)
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to {filename}")
        except IOError as e:
            print(f"Error saving results: {e}")
    
    def _generate_summary(self, results: Dict, analysis: Dict) -> Dict:
        """Generate a summary of the evaluation."""
        return {
            'total_tokenizers': len(results),
            'configurations_per_tokenizer': len(list(results.values())[0]) if results else 0,
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'best_overall_tokenizer': self._get_best_overall_tokenizer(analysis)
        }
    
    def _get_best_overall_tokenizer(self, analysis: Dict) -> Optional[str]:
        """Find the overall best tokenizer."""
        if not analysis:
            return None
        
        return max(analysis.items(), key=lambda x: x[1]['best_score'] if x[1]['best_score'] else -1)[0]
    
    def create_comparison_table(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Create a comparison table of all results.
        
        Args:
            results: Results dictionary from run_evaluation
            
        Returns:
            DataFrame with all results for easy comparison
        """
        rows = []
        
        for name, tk_results in results.items():
            for config_name, metrics in tk_results.items():
                row = self._create_comparison_row(name, config_name, metrics)
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _create_comparison_row(self, name: str, config: str, metrics: Dict) -> Dict:
        """Create a single row for the comparison table."""
        return {
            'Tokenizer': name,
            'Config': config,
            'Chunk_Size': metrics['chunk_size'],
            'Overlap': metrics['overlap'],
            'Total_Tokens': metrics['total_tokens'],
            'Avg_Tokens_Per_Chunk': metrics['avg_tokens_per_chunk'],
            'Std_Tokens_Per_Chunk': metrics['std_tokens_per_chunk'],
            'Compression_Ratio': metrics['compression_ratio'],
            'Vocab_Utilization': metrics['vocab_utilization'],
            'Avg_Encoding_Time_ms': metrics['avg_encoding_time'] * 1000
        }
    
    def save_comparison_table(self, results: Dict[str, Dict], filename: str = None) -> None:
        """Save comparison table to CSV.
        
        Args:
            results: Results dictionary from run_evaluation
            filename: Output filename (uses default if None)
        """
        filename = filename or OUTPUT_FILES['comparison']
        
        try:
            df = self.create_comparison_table(results)
            df.to_csv(filename, index=False)
            print(f"Comparison table saved to {filename}")
        except Exception as e:
            print(f"Error saving comparison table: {e}")

def main(text_file: str = "../data/sample_science_text.txt", 
         sizes: List[int] = None, 
         overlaps: List[int] = None) -> None:
    """Main function to run tokenizer evaluation.
    
    Args:
        text_file: Path to the text file to evaluate
        sizes: List of chunk sizes to test
        overlaps: List of overlaps to test
    """
    sizes = sizes or DEFAULT_CHUNK_SIZES
    overlaps = overlaps or DEFAULT_OVERLAPS
    
    print("Starting tokenizer evaluation...")
    print(f"Testing chunk sizes: {sizes}")
    print(f"Testing overlaps: {overlaps}")
    print("-" * 50)
    
    try:
        evaluator = TokenizerEvaluator()
        
        results = evaluator.run_evaluation(text_file, sizes, overlaps)
        
        if not results:
            print("No results generated. Check error messages above.")
            return
        
        analysis = evaluator.analyze_results(results)
        
        evaluator.save_results(results, analysis)
        evaluator.save_comparison_table(results)
        
        _print_evaluation_summary(analysis)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

def _print_evaluation_summary(analysis: Dict[str, Dict]) -> None:
    """Print a formatted summary of the evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    for name, data in analysis.items():
        if not data['best_metrics']:
            print(f"\n{name.upper()}: No valid results")
            continue
            
        best = data['best_metrics']
        print(f"\n{name.upper()}:")
        print(f"  Best Config: {data['best_config']}")
        print(f"  Chunk Size: {best['chunk_size']}, Overlap: {best['overlap']}")
        print(f"  Compression Ratio: {best['compression_ratio']:.2f}")
        print(f"  Avg Tokens/Chunk: {best['avg_tokens_per_chunk']:.1f}")
        print(f"  Std Tokens/Chunk: {best['std_tokens_per_chunk']:.1f}")
        print(f"  Avg Encoding Time: {best['avg_encoding_time']*1000:.2f}ms")
        print(f"  Vocab Utilization: {best['vocab_utilization']:.2%}")
        print(f"  Score: {data['best_score']:.4f}")
    
    valid = {k: v for k, v in analysis.items() if v['best_metrics']}
    if valid:
        best = max(valid.items(), key=lambda x: x[1]['best_score'])
        print(f"\n" + "=" * 60)
        print(f"OVERALL BEST: {best[0].upper()}")
        print(f"Best Config: {best[1]['best_config']}")
        print(f"Score: {best[1]['best_score']:.4f}")
        print("=" * 60)
    else:
        print("\nNo valid results found for any tokenizer.")

if __name__ == "__main__":
    import sys
    
    text_file = sys.argv[1] if len(sys.argv) > 1 else "../data/sample_science_text.txt"
    main(text_file)
