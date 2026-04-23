import PyPDF2

file_name = "iesc111"
reader = PyPDF2.PdfReader(f"../iesc1dd/{file_name}.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()
print(f"Extracted {len(text)} characters.")


with open(f"pypdf_{file_name}.txt", "w") as f:
    f.write(text)
