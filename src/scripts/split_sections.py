import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
extracted_dir = os.path.join(BASE_DIR, "extracted")

def split_text_file(filepath, pdf_name):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    section_pattern = re.compile(r"^(?:#*\s*)?(?:\*\*_*)?(\d+\.\d+(?:\.\d+)?)(?:_*\*\*)?\s*(.*)")
    exercise_header_pattern = re.compile(r"^(?:#*\s*)?(?:\*\*_*)?Exercises?(?:_*\*\*)?\s*$", re.IGNORECASE)

    example_pattern = re.compile(r"(?:^|[\n\r])(?:#*\s*)?(?:\*\*_*)?Example\s+\d+\.\d+(?:_*\*\*)?", re.IGNORECASE)
    solution_pattern = re.compile(r"(?:^|[\n\r])(?:#*\s*)?(?:\*\*_*)?Solution:?(?:_*\*\*)?", re.IGNORECASE)

    in_exercises = False
    in_example = False

    concepts_content = []
    exercises_content = []
    examples_content = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue

        if not in_exercises and exercise_header_pattern.search(stripped_line):
            in_exercises = True
            continue

        if not in_exercises:

            if example_pattern.search(stripped_line):
                in_example = True

            if in_example:
                examples_content.append(line)

            else:
                concepts_content.append(line)

            if section_pattern.match(stripped_line):
                in_example = False
        else:
            exercises_content.append(line)

    full_text = "".join(lines)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', full_text) if p.strip()]
    paragraphs_content = "\n\n".join(paragraphs)

    concepts_dir = os.path.join(extracted_dir, "concepts")
    exercises_dir = os.path.join(extracted_dir, "exercises")
    examples_dir = os.path.join(extracted_dir, "worked_examples")
    paragraphs_dir = os.path.join(extracted_dir, "paragraphs")

    os.makedirs(concepts_dir, exist_ok=True)
    os.makedirs(exercises_dir, exist_ok=True)
    os.makedirs(examples_dir, exist_ok=True)
    os.makedirs(paragraphs_dir, exist_ok=True)

    if concepts_content:
        file_name = f"{pdf_name}_concepts.txt"
        out_path = os.path.join(concepts_dir, file_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.writelines(concepts_content)

    if exercises_content:
        file_name = f"{pdf_name}_exercises.txt"
        out_path = os.path.join(exercises_dir, file_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.writelines(exercises_content)

    if examples_content:
        file_name = f"{pdf_name}_examples.txt"
        out_path = os.path.join(examples_dir, file_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.writelines(examples_content)

    if paragraphs_content:
        file_name = f"{pdf_name}_paragraphs.txt"
        out_path = os.path.join(paragraphs_dir, file_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(paragraphs_content)

    print(f"Processed {pdf_name}: Classified into concepts, exercises, examples, and paragraphs.")

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

