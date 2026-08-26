from PyPDF2 import PdfReader

def extract_text(file):
    text = ""

    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    else:
        text = file.read().decode("utf-8")

    return text