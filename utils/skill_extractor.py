def extract_skills(text):
    # Predefined skill set (can expand later)
    skills_db = {
        "python", "java", "c", "c++", "sql",
        "machine learning", "data structures",
        "system design", "react", "backend",
        "frontend", "django", "flask"
    }

    found_skills = set()

    for skill in skills_db:
        if skill in text:
            found_skills.add(skill)

    return found_skills