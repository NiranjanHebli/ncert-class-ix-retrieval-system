"""
Chunking Strategy Comparison for PariShiksha

This module compares the original fixed-size chunking approach with the new
semantic-aware chunking strategies, evaluating retrieval quality and performance.
"""

import os
import json
import time
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
import tiktoken
import re

from vec_retrieval import VectorDatabase
from improved_chunking import ImprovedVectorDatabase
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class RetrievalResult:
    """Results from a retrieval query"""
    query: str
    old_results: List[Dict[str, Any]]
    new_results: List[Dict[str, Any]]
    expected_category: str = None

@dataclass
class EvaluationResult:
    """Results from end-to-end evaluation"""
    question: str
    question_type: str
    old_answer: str
    new_answer: str
    old_retrieved_chunks: List[Dict[str, Any]]
    new_retrieved_chunks: List[Dict[str, Any]]
    correctness_old: str
    correctness_new: str
    grounded_old: str
    grounded_new: str
    refusal_old: str
    refusal_new: str

class ChunkingComparator:
    """
    Compare original vs improved chunking strategies.
    """

    def __init__(self):
        self.old_db = VectorDatabase(use_embeddings=False)
        self.new_db = ImprovedVectorDatabase()
        self.tiktoken_enc = tiktoken.get_encoding('cl100k_base')

    def setup_databases(
        self,
        extracted_dir: str = None,
        old_db_path: str = None,
        new_db_path: str = None,
    ):
        if extracted_dir is None: extracted_dir = str(PROJECT_ROOT / "extracted")
        if old_db_path is None: old_db_path = str(PROJECT_ROOT / "data/vector_db")
        if new_db_path is None: new_db_path = str(PROJECT_ROOT / "data/improved_vector_db")
        """Setup both databases for comparison"""
        if os.path.exists(old_db_path):
            self.old_db.load_from_disk(old_db_path)
        else:
            for root, dirs, files in os.walk(extracted_dir):
                if root == extracted_dir:
                    continue
                for f in files:
                    if f.endswith('.txt'):
                        self.old_db.build_chunk_store_from_file(
                            os.path.join(root, f)
                        )
            self.old_db.save_to_disk(old_db_path)

        if os.path.exists(new_db_path):
            self.new_db.load_from_disk(new_db_path)
        else:
            self.new_db.build_from_directory(extracted_dir)
            self.new_db.save_to_disk(new_db_path)

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comparative statistics for both databases"""
        old_stats = self._get_old_db_stats()
        new_stats = self.new_db.get_statistics()

        return {
            'original': old_stats,
            'improved': new_stats,
            'comparison': {
                'chunk_count_diff': new_stats['total_chunks'] - old_stats['total_chunks'],
                'avg_tokens_diff': new_stats['avg_tokens'] - old_stats['avg_tokens'],
                'over_limit_chunks': new_stats['chunks_over_limit']
            }
        }

    def _get_old_db_stats(self) -> Dict[str, Any]:
        """Get statistics for original database"""
        if not self.old_db.chunks:
            return {}

        type_counts = {}
        token_counts = []

        for chunk in self.old_db.chunks:
            ct = chunk['content_type']
            type_counts[ct] = type_counts.get(ct, 0) + 1
            token_counts.append(len(self.tiktoken_enc.encode(chunk['text'])))

        return {
            'total_chunks': len(self.old_db.chunks),
            'content_types': type_counts,
            'avg_tokens': np.mean(token_counts) if token_counts else 0,
            'max_tokens': max(token_counts) if token_counts else 0,
            'min_tokens': min(token_counts) if token_counts else 0,
            'chunks_over_limit': sum(1 for t in token_counts if t > 180)
        }

    def evaluate_retrieval_quality(
        self,
        test_queries: List[Dict[str, str]],
        k: int = 3,
    ) -> List[RetrievalResult]:
        """Evaluate retrieval quality on test queries"""
        results = []

        for query_data in test_queries:
            query = query_data['question']
            expected_category = query_data.get('category', 'unknown')

            old_results = self.old_db.retrieve_bm25(query, k=k)
            new_results = self.new_db.retrieve_bm25(query, k=k)

            result = RetrievalResult(
                query=query,
                old_results=old_results,
                new_results=new_results,
                expected_category=expected_category
            )
            results.append(result)

        return results

    def analyze_retrieval_overlap(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Analyze overlap between old and new retrieval results"""
        overlap_stats = {
            'total_queries': len(results),
            'exact_matches': 0,
            'partial_overlap': 0,
            'no_overlap': 0,
            'content_type_matches': 0,
            'semantic_boundary_analysis': {}
        }

        for result in results:
            old_chunks = set(r['chunk_id'] for r in result.old_results)
            new_chunks = set(r['chunk_id'] for r in result.new_results)

            intersection = old_chunks.intersection(new_chunks)
            if intersection:
                overlap_stats['exact_matches'] += 1
            elif old_chunks and new_chunks:
                overlap_stats['partial_overlap'] += 1
            else:
                overlap_stats['no_overlap'] += 1

            old_types = set(r['content_type'] for r in result.old_results)
            new_types = set(r['content_type'] for r in result.new_results)
            if old_types.intersection(new_types):
                overlap_stats['content_type_matches'] += 1

            for new_result in result.new_results:
                boundary = new_result.get('semantic_boundary', 'unknown')
                if boundary not in overlap_stats['semantic_boundary_analysis']:
                    overlap_stats['semantic_boundary_analysis'][boundary] = 0
                overlap_stats['semantic_boundary_analysis'][boundary] += 1

        return overlap_stats

    def _is_refusal(self, answer: str) -> bool:
        """Check if an answer is a refusal (indicates content not found)"""
        refusal_phrases = [
            'outside', 'not present', 'not in the context',
            'cannot answer', 'unable to answer', 'insufficient information'
        ]
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in refusal_phrases)

    def _evaluate_answer_quality(self, answer: str, question_type: str) -> Dict[str, str]:
        """Evaluate answer quality based on question type"""
        is_refusal = self._is_refusal(answer)

        if question_type == 'out_of_scope':
            correctness = 'yes' if is_refusal else 'no'
            grounded = 'yes' if is_refusal else 'no'
            refusal = 'yes' if is_refusal else 'no'
        else:
            correctness = 'yes' if not is_refusal and len(answer) > 20 else ('no' if is_refusal else 'partial')
            grounded = 'yes' if not is_refusal else 'no'
            refusal = 'na' if not is_refusal else 'no'

        return {
            'correctness': correctness,
            'grounded': grounded,
            'refusal': refusal
        }

    def generate_mock_answer(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Generate a mock answer based on retrieved chunks (simulates LLM behavior)"""
        if not retrieved_chunks:
            return "This question is outside the provided NCERT content."

        context = ' '.join([chunk['text'][:200] for chunk in retrieved_chunks])
        question_lower = question.lower()

        question_words = set(re.findall(r'\b\w+\b', question_lower))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        overlap = len(question_words.intersection(context_words))

        if overlap < 2:
            return "This question is outside the provided NCERT content."

        first_chunk = retrieved_chunks[0]['text']
        if len(first_chunk) > 100:
            return first_chunk[:150] + "..."
        else:
            return first_chunk

    def evaluate_end_to_end_retrieval(
        self,
        test_queries: List[Dict[str, str]],
        k: int = 3,
    ) -> List[EvaluationResult]:
        """Evaluate end-to-end retrieval quality similar to notebook approach"""
        results = []

        for query_data in test_queries:
            question = query_data['question']
            question_type = query_data.get('type', 'unknown')

            old_results = self.old_db.retrieve_bm25(question, k=k)
            new_results = self.new_db.retrieve_bm25(question, k=k)

            old_answer = self.generate_mock_answer(question, old_results)
            new_answer = self.generate_mock_answer(question, new_results)

            old_eval = self._evaluate_answer_quality(old_answer, question_type)
            new_eval = self._evaluate_answer_quality(new_answer, question_type)

            result = EvaluationResult(
                question=question,
                question_type=question_type,
                old_answer=old_answer,
                new_answer=new_answer,
                old_retrieved_chunks=old_results,
                new_retrieved_chunks=new_results,
                correctness_old=old_eval['correctness'],
                correctness_new=new_eval['correctness'],
                grounded_old=old_eval['grounded'],
                grounded_new=new_eval['grounded'],
                refusal_old=old_eval['refusal'],
                refusal_new=new_eval['refusal']
            )
            results.append(result)

        return results

    def measure_retrieval_speed(
        self,
        queries: List[str],
        iterations: int = 5,
    ) -> Dict[str, float]:
        """Measure retrieval speed for both approaches"""
        old_times = []
        new_times = []

        for _ in range(iterations):
            start_time = time.time()
            for query in queries:
                self.old_db.retrieve_bm25(query, k=3)
            old_times.append(time.time() - start_time)

            start_time = time.time()
            for query in queries:
                self.new_db.retrieve_bm25(query, k=3)
            new_times.append(time.time() - start_time)

        return {
            'original_avg_time': np.mean(old_times),
            'improved_avg_time': np.mean(new_times),
            'speed_ratio': np.mean(new_times) / np.mean(old_times) if np.mean(old_times) != 0 else 0
        }

    def calculate_evaluation_metrics(self, eval_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate evaluation metrics from end-to-end results"""
        metrics = {
            'total_questions': len(eval_results),
            'old_db': {
                'correct': 0,
                'grounded': 0,
                'proper_refusal': 0
            },
            'new_db': {
                'correct': 0,
                'grounded': 0,
                'proper_refusal': 0
            },
            'by_type': {}
        }

        for result in eval_results:
            q_type = result.question_type

            if q_type not in metrics['by_type']:
                metrics['by_type'][q_type] = {
                    'count': 0,
                    'old_correct': 0,
                    'new_correct': 0,
                    'old_grounded': 0,
                    'new_grounded': 0
                }

            metrics['by_type'][q_type]['count'] += 1

            if result.correctness_old == 'yes':
                metrics['old_db']['correct'] += 1
                metrics['by_type'][q_type]['old_correct'] += 1
            if result.grounded_old == 'yes':
                metrics['old_db']['grounded'] += 1
                metrics['by_type'][q_type]['old_grounded'] += 1
            if result.refusal_old == 'yes':
                metrics['old_db']['proper_refusal'] += 1

            if result.correctness_new == 'yes':
                metrics['new_db']['correct'] += 1
                metrics['by_type'][q_type]['new_correct'] += 1
            if result.grounded_new == 'yes':
                metrics['new_db']['grounded'] += 1
                metrics['by_type'][q_type]['new_grounded'] += 1
            if result.refusal_new == 'yes':
                metrics['new_db']['proper_refusal'] += 1

        total = metrics['total_questions']
        if total > 0:
            for db in ['old_db', 'new_db']:
                for metric in ['correct', 'grounded', 'proper_refusal']:
                    metrics[db][f'{metric}_pct'] = (metrics[db][metric] / total) * 100

            for q_type, type_metrics in metrics['by_type'].items():
                count = type_metrics['count']
                if count > 0:
                    for metric in ['old_correct', 'new_correct', 'old_grounded', 'new_grounded']:
                        type_metrics[f'{metric}_pct'] = (type_metrics[metric] / count) * 100

        return metrics

    def print_evaluation_results(self, eval_results: List[EvaluationResult]):
        """Print evaluation results in notebook-style format"""
        metrics = self.calculate_evaluation_metrics(eval_results)

        print("\n" + "=" * 80)
        print("END-TO-END RETRIEVAL EVALUATION RESULTS")
        print("=" * 80)

        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Questions: {metrics['total_questions']}")
        print(f"  Old DB - Correct: {metrics['old_db']['correct']}/{metrics['total_questions']} ({metrics['old_db']['correct_pct']:.1f}%)")
        print(f"  New DB - Correct: {metrics['new_db']['correct']}/{metrics['total_questions']} ({metrics['new_db']['correct_pct']:.1f}%)")
        print(f"  Old DB - Grounded: {metrics['old_db']['grounded']}/{metrics['total_questions']} ({metrics['old_db']['grounded_pct']:.1f}%)")
        print(f"  New DB - Grounded: {metrics['new_db']['grounded']}/{metrics['total_questions']} ({metrics['new_db']['grounded_pct']:.1f}%)")

        print(f"\nPERFORMANCE BY QUESTION TYPE:")
        for q_type, type_metrics in metrics['by_type'].items():
            print(f"  {q_type.upper()} ({type_metrics['count']} questions):")
            print(f"    Old Correct: {type_metrics['old_correct']}/{type_metrics['count']} ({type_metrics.get('old_correct_pct', 0):.1f}%)")
            print(f"    New Correct: {type_metrics['new_correct']}/{type_metrics['count']} ({type_metrics.get('new_correct_pct', 0):.1f}%)")
            print(f"    Old Grounded: {type_metrics['old_grounded']}/{type_metrics['count']} ({type_metrics.get('old_grounded_pct', 0):.1f}%)")
            print(f"    New Grounded: {type_metrics['new_grounded']}/{type_metrics['count']} ({type_metrics.get('new_grounded_pct', 0):.1f}%)")

        print(f"\nDETAILED RESULTS:")
        print(f'{"#":<3} {"Type":<15} {"Old_Correct":<12} {"New_Correct":<12} {"Old_Grounded":<12} {"New_Grounded":<12} {"Question":<50}')
        print('-' * 120)
        for i, result in enumerate(eval_results, 1):
            print(f'{i:<3} {result.question_type:<15} {result.correctness_old:<12} {result.correctness_new:<12} {result.grounded_old:<12} {result.grounded_new:<12} {result.question[:47]}...')

        print("\n" + "=" * 80)

    def save_evaluation_report(
        self,
        eval_results: List[EvaluationResult],
        output_path: str = None,
    ):
        if output_path is None: output_path = str(PROJECT_ROOT / "data/retrieval_evaluation_report.json")
        """Save evaluation report to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        metrics = self.calculate_evaluation_metrics(eval_results)

        serializable_results = []
        for result in eval_results:
            serializable_results.append({
                'question': result.question,
                'question_type': result.question_type,
                'old_answer': result.old_answer,
                'new_answer': result.new_answer,
                'correctness_old': result.correctness_old,
                'correctness_new': result.correctness_new,
                'grounded_old': result.grounded_old,
                'grounded_new': result.grounded_new,
                'refusal_old': result.refusal_old,
                'refusal_new': result.refusal_new,
                'old_chunk_count': len(result.old_retrieved_chunks),
                'new_chunk_count': len(result.new_retrieved_chunks)
            })

        report = {
            'metrics': metrics,
            'detailed_results': serializable_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    def generate_comparison_report(self, test_queries: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        stats = self.get_database_statistics()

        retrieval_results = self.evaluate_retrieval_quality(test_queries)

        overlap_analysis = self.analyze_retrieval_overlap(retrieval_results)

        sample_queries = [q['question'] for q in test_queries[:10]]
        speed_stats = self.measure_retrieval_speed(sample_queries)

        quality_metrics = self._calculate_quality_metrics(retrieval_results)

        report = {
            'database_statistics': stats,
            'overlap_analysis': overlap_analysis,
            'speed_performance': speed_stats,
            'quality_metrics': quality_metrics
        }

        return report

    def _calculate_quality_metrics(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Calculate quality metrics for retrieval results"""
        metrics = {
            'avg_relevance_score_old': 0,
            'avg_relevance_score_new': 0,
            'content_type_relevance': {},
            'semantic_boundary_effectiveness': {}
        }

        old_scores = []
        new_scores = []

        for result in results:
            old_query_scores = [r.get('score', 0) for r in result.old_results]
            new_query_scores = [r.get('score', 0) for r in result.new_results]

            old_scores.extend(old_query_scores)
            new_scores.extend(new_query_scores)

            for new_result in result.new_results:
                content_type = new_result['content_type']
                boundary = new_result['semantic_boundary']

                if content_type not in metrics['content_type_relevance']:
                    metrics['content_type_relevance'][content_type] = []
                metrics['content_type_relevance'][content_type].append(new_result['score'])

                if boundary not in metrics['semantic_boundary_effectiveness']:
                    metrics['semantic_boundary_effectiveness'][boundary] = []
                metrics['semantic_boundary_effectiveness'][boundary].append(new_result['score'])

        metrics['avg_relevance_score_old'] = np.mean(old_scores) if old_scores else 0
        metrics['avg_relevance_score_new'] = np.mean(new_scores) if new_scores else 0

        for content_type in metrics['content_type_relevance']:
            scores = metrics['content_type_relevance'][content_type]
            metrics['content_type_relevance'][content_type] = np.mean(scores) if scores else 0

        for boundary in metrics['semantic_boundary_effectiveness']:
            scores = metrics['semantic_boundary_effectiveness'][boundary]
            metrics['semantic_boundary_effectiveness'][boundary] = np.mean(scores) if scores else 0

        return metrics

    def _generate_recommendations(self, stats: Dict, overlap: Dict, quality: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        return []

    def save_comparison_report(
        self,
        report: Dict[str, Any],
        output_path: str = None,
    ):
        if output_path is None: output_path = str(PROJECT_ROOT / "data/chunking_comparison_report.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    def print_summary_report(self, report: Dict[str, Any]):
        """Print a summary of the comparison report"""
        print("\n" + "=" * 60)
        print("CHUNKING STRATEGY COMPARISON REPORT")
        print("=" * 60)

        stats = report['database_statistics']
        print(f"\nDATABASE STATISTICS:")
        print(f"  Original chunks: {stats['original']['total_chunks']}")
        print(f"  Improved chunks: {stats['improved']['total_chunks']}")
        print(f"  Difference: {stats['comparison']['chunk_count_diff']}")
        print(f"  Avg tokens (old): {stats['original']['avg_tokens']:.1f}")
        print(f"  Avg tokens (new): {stats['improved']['avg_tokens']:.1f}")
        print(f"  Chunks over limit: {stats['improved']['chunks_over_limit']}")

        print(f"\nCONTENT TYPE DISTRIBUTION (Improved):")
        for content_type, count in stats['improved']['content_types'].items():
            print(f"  {content_type}: {count} chunks")

        print(f"\nSEMANTIC BOUNDARIES:")
        for boundary, count in stats['improved']['semantic_boundaries'].items():
            print(f"  {boundary}: {count} chunks")

        overlap = report['overlap_analysis']
        print(f"\nRETRIEVAL OVERLAP ANALYSIS:")
        print(f"  Total queries: {overlap['total_queries']}")
        if overlap['total_queries'] > 0:
            print(f"  Exact matches: {overlap['exact_matches']} ({overlap['exact_matches']/overlap['total_queries']*100:.1f}%)")
            print(f"  Partial overlap: {overlap['partial_overlap']} ({overlap['partial_overlap']/overlap['total_queries']*100:.1f}%)")
            print(f"  No overlap: {overlap['no_overlap']} ({overlap['no_overlap']/overlap['total_queries']*100:.1f}%)")

        quality = report['quality_metrics']
        print(f"\nQUALITY METRICS:")
        print(f"  Avg relevance (old): {quality['avg_relevance_score_old']:.3f}")
        print(f"  Avg relevance (new): {quality['avg_relevance_score_new']:.3f}")
        print(f"  Improvement: {quality['avg_relevance_score_new'] - quality['avg_relevance_score_old']:+.3f}")

        speed = report['speed_performance']
        print(f"\nSPEED PERFORMANCE:")
        print(f"  Old avg time: {speed['original_avg_time']:.3f}s")
        print(f"  New avg time: {speed['improved_avg_time']:.3f}s")
        print(f"  Speed ratio: {speed['speed_ratio']:.2f}x")

        print("\n" + "=" * 60)

def run_comparison():
    """Run the complete comparison analysis"""
    comparator = ChunkingComparator()

    comparator.setup_databases()

    with open(str(PROJECT_ROOT / 'data/eval_questions.json'), 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_queries = []
    for category in test_data:
        for question in category['questions']:
            test_queries.append({
                'question': question,
                'category': category['type']
            })

    report = comparator.generate_comparison_report(test_queries)

    comparator.save_comparison_report(report)
    comparator.print_summary_report(report)

    return report

def run_retrieval_evaluation():
    """Run end-to-end retrieval evaluation similar to notebook approach"""
    comparator = ChunkingComparator()

    comparator.setup_databases()

    with open('data/eval_questions.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_queries = []
    for category in test_data:
        for question in category['questions']:
            test_queries.append({
                'question': question,
                'type': category['type']
            })

    eval_results = comparator.evaluate_end_to_end_retrieval(test_queries)

    comparator.print_evaluation_results(eval_results)
    comparator.save_evaluation_report(eval_results)

    return eval_results

if __name__ == "__main__":
    run_comparison()
    run_retrieval_evaluation()

