import pymupdf4llm

print("Extracting with pymupdf4llm...")
try:
    file_name = "iesc111"
    md_text = pymupdf4llm.to_markdown(f"../iesc1dd/{file_name}.pdf")
    print(md_text[1000:2000])
    
    with open(f"scratch_pymupdf4llm_{file_name}.txt", "w") as f:
        f.write(md_text)
except Exception as e:
    print(e)

    
