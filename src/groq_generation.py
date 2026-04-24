import os
import sys
import json
from groq import Groq
from dotenv import load_dotenv
from vec_retrieval import VectorDatabase

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Grounding prompt (Consistent with Stage 3 requirements)
GROUNDING_PROMPT = """
You are a study assistant for PariShiksha. 
Use ONLY the context provided below to answer the question.
If the answer is not present in the context, respond with:
"This question is outside the provided NCERT content."
Do not infer, extrapolate, or use outside knowledge.

Context:
{context}

Question: {question}
Answer:
"""

class GroqGroundedGenerator:
    """Modular class for grounded generation using Groq (Llama 3)."""
    
    def __init__(self, model_name="llama-3.1-8b-instant"):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in .env file.")
        
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model_name = model_name
        self.db = VectorDatabase(use_embeddings=False)
        
    def initialize_db(self, text_file):
        """Build chunk store from a specific text file."""
        self.db.build_chunk_store_from_file(text_file)

    def answer(self, question, k=3):
        """
        Retrieves context and generates a grounded answer using Groq.
        """
        retrieved_chunks = self.db.retrieve_bm25(question, k=k)
        context = "\n\n---\n\n".join([chunk['text'] for chunk in retrieved_chunks])
        prompt = GROUNDING_PROMPT.format(context=context, question=question)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return {"answer": completion.choices[0].message.content, "retrieved_chunks": retrieved_chunks}
        except Exception as e:
            return {"error": str(e), "retrieved_chunks": retrieved_chunks}

def run_demo():
    """Standard test on Chapter 1."""
    print("\n--- Groq Grounded Generation Demo (Chapter 1) ---")
    generator = GroqGroundedGenerator()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_file = os.path.join(project_root, "extracted", "iesc101.txt")
    
    if os.path.exists(sample_file):
        generator.initialize_db(sample_file)
        q = "What is the physical nature of matter?"
        print(f"Question: {q}")
        print(f"Answer: {generator.answer(q)['answer']}")
    else:
        print("Error: extracted/iesc101.txt not found.")

def run_full_corpus_demo():
    """Demo across all chapters."""
    print("\n--- Groq Full Corpus Demo (All Chapters) ---")
    generator = GroqGroundedGenerator()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extracted_dir = os.path.join(project_root, "extracted")
    
    files_loaded = 0
    for root, dirs, files in os.walk(extracted_dir):
        if root == extracted_dir: continue
        for f in files:
            if f.endswith(".txt"):
                generator.initialize_db(os.path.join(root, f))
                files_loaded += 1
    
    print(f"Loaded {files_loaded} files.")
    
    test_questions = []
    questions_file = os.path.join(project_root, "data", "eval_questions.json")
    if os.path.exists(questions_file):
        with open(questions_file, "r") as f:
            categories = json.load(f)
            for cat in categories:
                test_questions.extend(cat["questions"])
    else:
        # Fallback if file missing
        test_questions = ["State the universal law of gravitation."]
    
    for q in test_questions:
        print(f"\n[QUESTION]: {q}")
        res = generator.answer(q)
        print(f"[ANSWER]: {res['answer']}")

def run_interactive_session():
    """Interactive session for the user."""
    print("\n--- Interactive Q&A Mode (Groq) ---")
    generator = GroqGroundedGenerator()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extracted_dir = os.path.join(project_root, "extracted")
    
    for root, dirs, files in os.walk(extracted_dir):
        if root == extracted_dir: continue
        for f in files:
            if f.endswith(".txt"):
                generator.initialize_db(os.path.join(root, f))
    
    print("System Ready! Type 'exit' to quit.")
    while True:
        q = input("\n[USER]: ").strip()
        if q.lower() in ['exit', 'quit']: break
        if not q: continue
        print(f"[ASSISTANT]: {generator.answer(q)['answer']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--full": run_full_corpus_demo()
        elif sys.argv[1] == "--interactive": run_interactive_session()
    else:
        run_demo()
