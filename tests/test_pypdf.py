import PyPDF2
reader = PyPDF2.PdfReader("iesc1dd/iesc104.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()
print(f"Extracted {len(text)} characters.")
print(text[:200])
