import spacy
from utils.preprocessing import clean_text

class SkillExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def extract_skills(self, text, skill_database=None):
        if skill_database is None:
            skill_database = ["python", "sql", "machine learning", "data analysis", "pandas", "numpy", "power bi", "excel", "tableau"]
        
        cleaned = clean_text(text)
        text_lower = text.lower()
        extracted = set()
        
        for skill in skill_database:
            if skill.lower() in text_lower:
                extracted.add(skill)
                
        return list(extracted)

    def extract_skills_semantic(self, text, skill_database=None):
        # app.py is calling this method
        return self.extract_skills(text, skill_database)