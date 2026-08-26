import os
import json
import math
from google import genai
from google.genai import types

def cosine_similarity(vec1, vec2):
    """Legacy TF-IDF Vector Cosine Similarity."""
    dot_product = 0
    norm1 = 0
    norm2 = 0

    all_words = set(vec1.keys()).union(set(vec2.keys()))

    for word in all_words:
        v1 = vec1.get(word, 0)
        v2 = vec2.get(word, 0)

        dot_product += v1 * v2
        norm1 += v1 ** 2
        norm2 += v2 ** 2

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))


def cot_llm_ats_score(resume_text: str, jd_text: str, api_key: str = None) -> dict:
    """
    Evaluates candidate fit using Chain-of-Thought (CoT) Prompting.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {
            "thought_process": "Gemini API Key missing. Unable to perform CoT analysis.",
            "ats_score": 0,
            "matching_skills": [],
            "missing_skills": []
        }

    client = genai.Client(api_key=api_key)

    prompt = f"""You are an ATS (Applicant Tracking System) Evaluation Engine. Analyze the candidate's resume against the Job Description step-by-step.

Use the following Chain-of-Thought reasoning structure before deciding on the final score:

Step 1: Extract and list key requirements (skills, experience, responsibilities) from the Job Description.
Step 2: Cross-reference the Candidate's Resume to identify directly matched skills and qualified experiences.
Step 3: Identify critical missing skills, missing domain knowledge, or gaps in experience.
Step 4: Formulate a final ATS Match Score (0 to 100%) reflecting modern technical recruitment criteria.

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

CANDIDATE RESUME:
\"\"\"{resume_text}\"\"\"

Return your response in ONLY valid JSON with these exact keys:
{{
  "thought_process": "Step 1: ... Step 2: ... Step 3: ... Step 4: ...",
  "ats_score": 85,
  "matching_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3", "skill4"]
}}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {
            "thought_process": f"Error performing CoT Evaluation: {str(e)}",
            "ats_score": 0,
            "matching_skills": [],
            "missing_skills": []
        }