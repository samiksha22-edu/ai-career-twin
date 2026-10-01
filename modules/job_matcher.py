import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class JobMatcher:
    def __init__(self, jobs_csv: str = "data/jobs.csv"):
        self.df = pd.read_csv(jobs_csv)

    def match_jobs(self, user_skills: list) -> list:
        """Calculates percentage matching score across all job categories."""
        user_str = " ".join([s.replace(" ", "") for s in user_skills])
        job_profiles = self.df.groupby('job_role')['skills'].apply(lambda x: " ".join(x)).reset_index()
        
        job_profiles['formatted_skills'] = job_profiles['skills'].apply(
            lambda x: " ".join([s.strip().replace(" ", "") for s in x.split(',')])
        )

        all_texts = [user_str] + job_profiles['formatted_skills'].tolist()
        cv = CountVectorizer().fit_transform(all_texts)
        vectors = cv.toarray()

        user_vec = vectors[0].reshape(1, -1)
        job_vecs = vectors[1:]

        sims = cosine_similarity(user_vec, job_vecs)[0]

        results = []
        for idx, row in job_profiles.iterrows():
            results.append({
                "job_role": row['job_role'],
                "match_score": round(float(sims[idx]) * 100, 2)
            })

        return sorted(results, key=lambda x: x['match_score'], reverse=True)