from utils.text_extractor import extract_text

# Path to your resume file
file_path = "data/sample_resume.txt"

# Extract text
text = extract_text(file_path)

# Print output
print("----- Resume Content -----\n")
print(text)