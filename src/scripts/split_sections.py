import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
extracted_dir = os.path.join(BASE_DIR, "extracted")

def split_text_file(filepath, pdf_name):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Regular expressions for identifying sections and exercises
    section_pattern = re.compile(r"^(?:#*\s*)?(?:\*\*_*)?(\d+\.\d+(?:\.\d+)?)(?:_*\*\*)?\s*(.*)")
    exercise_header_pattern = re.compile(r"^(?:#*\s*)?(?:\*\*_*)?Exercises?(?:_*\*\*)?\s*$", re.IGNORECASE)
    exercise_item_pattern = re.compile(r"^(?:#*\s*)?(?:\*\*_*)?(\d+)\.(?!\d)\s*(.*)")
    
    in_exercises = False
    
    sections_content = []
    exercises_content = []
    
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
            
        if not in_exercises and exercise_header_pattern.search(stripped_line):
            in_exercises = True
            continue
            
        if not in_exercises:
            sections_content.append(line)
        else:
            exercises_content.append(line)

    # Paragraph extraction
    full_text = "".join(lines)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', full_text) if p.strip()]
    paragraphs_content = "\n\n".join(paragraphs)

    # Output directories
    sections_dir = os.path.join(extracted_dir, "sections")
    exercises_dir = os.path.join(extracted_dir, "exercises")
    paragraphs_dir = os.path.join(extracted_dir, "paragraphs")
    
    os.makedirs(sections_dir, exist_ok=True)
    os.makedirs(exercises_dir, exist_ok=True)
    os.makedirs(paragraphs_dir, exist_ok=True)
    
    # Write sections file
    if sections_content:
        file_name = f"{pdf_name}_sections.txt"
        out_path = os.path.join(sections_dir, file_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.writelines(sections_content)
            
    # Write exercises file
    if exercises_content:
        file_name = f"{pdf_name}_exercises.txt"
        out_path = os.path.join(exercises_dir, file_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.writelines(exercises_content)

    # Write paragraphs file
    if paragraphs_content:
        file_name = f"{pdf_name}_paragraphs.txt"
        out_path = os.path.join(paragraphs_dir, file_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(paragraphs_content)

    print(f"Processed {pdf_name}: Consolidated sections, exercises, and paragraphs.")

def main():
    if not os.path.exists(extracted_dir):
        print(f"Error: Directory not found at {extracted_dir}")
        return
        
    for filename in os.listdir(extracted_dir):
        if not filename.lower().endswith(".txt"):
            continue
            
        filepath = os.path.join(extracted_dir, filename)
        pdf_name = os.path.splitext(filename)[0]
        
        if os.path.isdir(filepath):
            continue
            
        split_text_file(filepath, pdf_name)

if __name__ == "__main__":
    main()
