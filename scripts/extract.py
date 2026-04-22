import requests
import os
import PyPDF2
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
url = "https://api.ocr.space/parse/image"
FOLDER_NAME  = "iesc1dd"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
source_dir = os.path.join(BASE_DIR, FOLDER_NAME)
output_dir = os.path.join(BASE_DIR, "docs")

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
    
    if os.path.exists(output_path):
        print(f"Skipping {filename}, already processed.")
        continue

    print(f"Processing {filename}...")
    
    with open(file_path, "rb") as f:
        try:
            response = requests.post(
                url,
                files={"file": f},
                data={
                    "apikey": API_KEY,  
                    "language": "eng"
                }
            )
        except Exception as e:
            print(f"Request failed for {filename}: {e}")
            continue

    if response.status_code != 200:
        print(f"Error: API returned status code {response.status_code} for {filename}")
        continue

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        print(f"Error: Could not decode JSON response for {filename}.")
        continue

    if result.get("IsErroredOnProcessing"):
        error_msgs = result.get("ErrorMessage", [""])
        error_msg = error_msgs[0] if isinstance(error_msgs, list) and len(error_msgs) > 0 else str(error_msgs)
        print(f"API Error processing {filename}:", error_msgs)
        
        # Fallback for file size limit error
        if "size exceeds" in error_msg.lower() or "1024 kb" in error_msg.lower():
            print(f"Falling back to local PyPDF2 extraction for {filename} due to size limit...")
            try:
                reader = PyPDF2.PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                with open(output_path, "w", encoding="utf-8") as out_f:
                    out_f.write(text)
                print(f"Saved text to {output_filename} using PyPDF2 fallback")
            except Exception as e:
                print(f"Fallback extraction failed for {filename}: {e}")
        
        continue

    try:
        parsed_results = result.get("ParsedResults")
        if not parsed_results:
            print(f"No text found in {filename}.")
            continue
            
        text = parsed_results[0].get("ParsedText", "")
        
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write(text)
            
        print(f"Saved text to {output_filename}")
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        
    import time
    time.sleep(2)  # Avoid rate limiting