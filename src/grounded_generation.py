import os
import sys
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from vec_retrieval import VectorDatabase

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Stage 3.1: Design a strong grounding prompt
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

class GroundedGenerator:
    """Modular class for performing grounded retrieval-augmented generation."""
    
    def __init__(self, model_name="gemini-2.5-flash"):
        if not API_KEY:
            raise ValueError("API_KEY not found in .env file.")
        
        self.client = genai.Client(api_key=API_KEY)
        self.model_name = model_name
        self.db = VectorDatabase(use_embeddings=False)
        
        # Simple model verification
        try:
            models = [m.name for m in self.client.models.list()]
            if not any(self.model_name in m for m in models):
                # Fallback to something likely available if 1.5-flash is missing
                available_flash = [m for m in models if "flash" in m]
                if available_flash:
                    self.model_name = available_flash[0]
        except Exception:
            pass

    def initialize_db(self, text_file):
        """Build chunk store from a specific text file."""
        self.db.build_chunk_store_from_file(text_file)

    def save_db(self, path="data/vector_db"):
        """Save current database to disk."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(project_root, path) if not os.path.isabs(path) else path
        self.db.save_to_disk(full_path)

    def load_db(self, path="data/vector_db"):
        """Load database from disk."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(project_root, path) if not os.path.isabs(path) else path
        if os.path.exists(full_path):
            self.db.load_from_disk(full_path)
            return True
        return False

    def answer(self, question, k=3):
        """
        Stage 3.2: Implement answer() function
        Retrieves context and generates a grounded answer using the new google-genai SDK.
        """
        retrieved_chunks = self.db.retrieve_bm25(question, k=k)
        context = "\n\n---\n\n".join([chunk['text'] for chunk in retrieved_chunks])
        prompt = GROUNDING_PROMPT.format(context=context, question=question)
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            return {"answer": response.text, "retrieved_chunks": retrieved_chunks}
        except Exception as e:
            return {"error": str(e), "retrieved_chunks": retrieved_chunks}

def run_demo():
    """Standard test on Chapter 1."""
    print("\n--- Grounded Generation Demo (Chapter 1) ---")
    generator = GroundedGenerator()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_file = os.path.join(project_root, "extracted", "iesc101.txt")
    
    if os.path.exists(sample_file):
        generator.initialize_db(sample_file)
        q = "What are the characteristics of particles of matter?"
        print(f"Question: {q}")
        print(f"Answer: {generator.answer(q)['answer']}")
    else:
        print("Error: extracted/iesc101.txt not found.")

def run_full_corpus_demo():
    """Demo across all chapters."""
    print("\n--- Full Corpus Demo (All Chapters) ---")
    generator = GroundedGenerator()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "data", "vector_db")
    extracted_dir = os.path.join(project_root, "extracted")
    
    if generator.load_db(db_path):
        print("Loaded existing database from disk.")
    else:
        print("Building new database from scratch...")
        files_loaded = 0
        for root, dirs, files in os.walk(extracted_dir):
            if root == extracted_dir: continue
            for f in files:
                if f.endswith(".txt"):
                    generator.initialize_db(os.path.join(root, f))
                    files_loaded += 1
        print(f"Loaded {files_loaded} files.")
        generator.save_db(db_path)
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
    print("\n--- Interactive Q&A Mode (Gemini) ---")
    generator = GroundedGenerator()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "data", "vector_db")
    extracted_dir = os.path.join(project_root, "extracted")
    
    if generator.load_db(db_path):
        print("Loaded existing database from disk.")
    else:
        print("Building new database from scratch...")
        for root, dirs, files in os.walk(extracted_dir):
            if root == extracted_dir: continue
            for f in files:
                if f.endswith(".txt"):
                    generator.initialize_db(os.path.join(root, f))
        generator.save_db(db_path)
    
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
