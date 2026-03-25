def extract_text(file_path):
    """
    Function to read text from a .txt file
    """

    text = ""

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        return text

    except FileNotFoundError:
        return "Error: File not found."

    except Exception as e:
        return f"Error: {e}"