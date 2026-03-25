from utils.text_extractor import extract_text
from utils.text_preprocessor import preprocess_text

file_path = "data/sample_resume.txt"

# Step 1: Extract text
text = extract_text(file_path)

print("----- Original Text -----\n")
print(text)

# Step 2: Clean text
clean_text = preprocess_text(text)

print("\n----- Cleaned Text -----\n")
print(clean_text)