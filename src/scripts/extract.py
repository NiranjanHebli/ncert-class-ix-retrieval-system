import os
import pymupdf4llm
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FOLDER_NAME  = "iesc1dd"
source_dir = os.path.join(BASE_DIR, FOLDER_NAME)
output_dir = os.path.join(BASE_DIR, "extracted")

if os.path.exists(output_dir):
    print(f"Cleaning up {output_dir} for a fresh run...")

    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(source_dir):
    print(f"Error: Directory not found at {source_dir}")
    exit(1)

for filename in os.listdir(source_dir):
    if not filename.lower().endswith(".pdf"):
        continue

    file_path = os.path.join(source_dir, filename)
    output_filename = os.path.splitext(filename)[0] + ".txt"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Processing {filename} with pymupdf4llm...")

    try:

        md_text = pymupdf4llm.to_markdown(file_path)

        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write(md_text)

        print(f"Saved formatted markdown text to {output_filename}")
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")

print("\nStarting section and exercise extraction...")
os.system(f"python3 {os.path.join(BASE_DIR, 'src', 'scripts', 'split_sections.py')}")
