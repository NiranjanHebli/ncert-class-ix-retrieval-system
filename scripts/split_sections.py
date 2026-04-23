import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs_dir = os.path.join(BASE_DIR, "docs")

def split_text_file(filepath, pdf_name):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Regular expressions for identifying sections and exercises
    # Matches markdown headers like `## **8.1 Balanced and Unbalanced Forces**` or just `1.1 Intro`
    section_pattern = re.compile(r"^(?:#*\s*)?(?:\*\*_*)?(\d+\.\d+(?:\.\d+)?)(?:_*\*\*)?\s*(.*)")
    # Match "Exercises" even with markdown bold/italic
    exercise_header_pattern = re.compile(r"^(?:#*\s*)?(?:\*\*_*)?Exercises?(?:_*\*\*)?\s*$", re.IGNORECASE)
    # Match "1. An object"
    exercise_item_pattern = re.compile(r"^(?:#*\s*)?(?:\*\*_*)?(\d+)\.(?!\d)\s*(.*)")
    
    current_section = "Intro"
    in_exercises = False
    current_exercise = "Intro"
    
    # Store lines for each section and exercise
    sections = {"Intro": []}
    exercises = {"Intro": []}
    
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
            
        # Detect start of exercises
        if not in_exercises and exercise_header_pattern.search(stripped_line):
            in_exercises = True
            continue
            
        if not in_exercises:
            # Check if this line actually looks like an exercise even before header
            ex_match = exercise_item_pattern.match(stripped_line)
            sec_match = section_pattern.match(stripped_line)
            
            # Heuristic: if we see a numbered list starting with 1. near the end, it might be exercises
            # But it's safer to just rely on section match
            if sec_match and not sec_match.group(2).startswith("is called"):
                current_section = sec_match.group(1)
                if current_section not in sections:
                    sections[current_section] = []
            sections[current_section].append(line)
        else:
            ex_match = exercise_item_pattern.match(stripped_line)
            if ex_match:
                current_exercise = ex_match.group(1)
                if current_exercise not in exercises:
                    exercises[current_exercise] = []
            exercises[current_exercise].append(line)

    # Output directory for the PDF
    # "folder name of the pdf file"
    output_dir = os.path.join(docs_dir, pdf_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Write sections
    for sec_num, sec_lines in sections.items():
        if sec_num == "Intro" and not sec_lines:
            continue
        file_name = f"{pdf_name}_section_{sec_num}.txt"
        out_path = os.path.join(output_dir, file_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.writelines(sec_lines)
            
    # Write exercises
    for ex_num, ex_lines in exercises.items():
        if ex_num == "Intro" and not ex_lines:
            continue
        file_name = f"{pdf_name}_exercise_{ex_num}.txt"
        out_path = os.path.join(output_dir, file_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.writelines(ex_lines)

    print(f"Processed {pdf_name}: {len(sections)} sections, {len(exercises)} exercises.")

def main():
    if not os.path.exists(docs_dir):
        print(f"Error: Directory not found at {docs_dir}")
        return
        
    for filename in os.listdir(docs_dir):
        if not filename.lower().endswith(".txt"):
            continue
            
        filepath = os.path.join(docs_dir, filename)
        pdf_name = os.path.splitext(filename)[0]
        
        # Avoid processing directories or already processed output
        if os.path.isdir(filepath):
            continue
            
        split_text_file(filepath, pdf_name)

if __name__ == "__main__":
    main()
