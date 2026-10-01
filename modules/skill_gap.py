import pandas as pd

class SkillGapAnalyzer:
    def __init__(self, jobs_csv: str = "data/jobs.csv"):
        self.df = pd.read_csv(jobs_csv)

    def get_role_skills(self, target_role: str) -> set:
        """Fetches normalized target skills for a given job role."""
        role_df = self.df[self.df['job_role'].str.lower() == target_role.lower()]
        if role_df.empty:
            return set()
        
        all_skills = []
        for raw in role_df['skills']:
            all_skills.extend([s.strip() for s in raw.split(',')])
        
        series = pd.Series(all_skills).value_counts()
        threshold = max(1, int(len(role_df) * 0.3))
        return set(series[series >= threshold].index)

    def analyze_gap(self, user_skills: list, target_role: str) -> dict:
        """Analyzes skill coverage, calculates readiness percentage, and assigns gap priority."""
        target_skills = self.get_role_skills(target_role)
        if not target_skills:
            return {"error": f"Role '{target_role}' not found in database."}

        user_skills_set = set(user_skills)
        matched_skills = user_skills_set.intersection(target_skills)
        missing_skills = list(target_skills - user_skills_set)

        readiness = (len(matched_skills) / len(target_skills)) * 100 if target_skills else 0

        prioritized = []
        for skill in missing_skills:
            if skill in ["SQL", "Machine Learning", "Python", "AWS", "Java", "Statistics"]:
                priority = "High"
            elif skill in ["TensorFlow", "Power BI", "Docker", "PyTorch", "Tableau", "React"]:
                priority = "Medium"
            else:
                priority = "Low"
            prioritized.append({"skill": skill, "priority": priority})

        p_order = {"High": 0, "Medium": 1, "Low": 2}
        prioritized.sort(key=lambda x: p_order[x["priority"]])

        return {
            "target_role": target_role,
            "readiness_score": round(readiness, 2),
            "matched_skills": list(matched_skills),
            "missing_skills": prioritized
        }