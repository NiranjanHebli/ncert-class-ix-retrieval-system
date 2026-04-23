import pymupdf4llm

print("Extracting with pymupdf4llm...")
try:
    md_text = pymupdf4llm.to_markdown("iesc1dd/iesc108.pdf")
    print(md_text[1000:2000])
    
    with open("scratch_pymupdf4llm.txt", "w") as f:
        f.write(md_text)
except Exception as e:
    print(e)
