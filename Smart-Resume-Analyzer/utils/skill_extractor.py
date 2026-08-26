import os
import json
from google import genai
from google.genai import types

def extract_skills_with_llm(text: str, api_key: str = None) -> list:
    """
    Extracts explicit and implicit skills using Few-Shot Prompting.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Fallback keyword matching if no API key is set
        db = {"python", "java", "c", "c++", "sql", "machine learning", "data structures", 
              "system design", "react", "backend", "frontend", "django", "flask", "aws", "mongodb"}
        return [skill for skill in db if skill in text.lower()]

    client = genai.Client(api_key=api_key)

    prompt = f"""You are an expert resume parser. Extract all technical skills, programming languages, frameworks, tools, and key soft skills from the provided text.

--- FEW-SHOT EXAMPLES ---

Input: "Engineered scalable REST services using Go and PostgreSQL. Led agile sprint reviews with 4 engineers."
Output: {{"skills": ["Go", "PostgreSQL", "REST APIs", "Agile", "Sprint Planning", "Leadership"]}}

Input: "Analyzed business metrics in Excel, built interactive Dash dashboards, and deployed to AWS EC2."
Output: {{"skills": ["Data Analysis", "Microsoft Excel", "Dash", "Python", "AWS", "AWS EC2"]}}

--- END OF EXAMPLES ---

Target Text:
\"\"\"{text}\"\"\"

Return ONLY valid JSON matching this schema:
{{"skills": ["skill1", "skill2"]}}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text)
        return data.get("skills", [])
    except Exception as e:
        print(f"Error in extract_skills_with_llm: {e}")
        # Return fallback
        db = {"python", "java", "c", "c++", "sql", "machine learning", "data structures", "system design", "react", "backend", "frontend", "django", "flask"}
        return [skill for skill in db if skill in text.lower()]