class RoadmapGenerator:
    def generate_roadmap(self, missing_skills: list) -> list:
        """Maps missing skill lists into structured monthly learning tracks."""
        if not missing_skills:
            return [{"month": "Month 1", "focus": "Interview Preparation & Advanced Projects", "level": "Advanced"}]

        high_p = [s['skill'] for s in missing_skills if s['priority'] == 'High']
        med_p = [s['skill'] for s in missing_skills if s['priority'] == 'Medium']
        low_p = [s['skill'] for s in missing_skills if s['priority'] == 'Low']

        roadmap = []
        month = 1

        if high_p:
            roadmap.append({
                "month": f"Month {month}",
                "focus": f"Core Mastery: {', '.join(high_p)}",
                "level": "Beginner to Intermediate"
            })
            month += 1

        if med_p:
            roadmap.append({
                "month": f"Month {month}",
                "focus": f"Frameworks & Applied Tools: {', '.join(med_p)}",
                "level": "Intermediate"
            })
            month += 1

        if low_p:
            roadmap.append({
                "month": f"Month {month}",
                "focus": f"Tooling & Supporting Technologies: {', '.join(low_p)}",
                "level": "Intermediate to Advanced"
            })
            month += 1

        roadmap.append({
            "month": f"Month {month}",
            "focus": "Capstone Portfolio Project & Real-World Application",
            "level": "Advanced"
        })
        roadmap.append({
            "month": f"Month {month + 1}",
            "focus": "System Design, Mock Interviews & Resume Optimization",
            "level": "Job Ready"
        })

        return roadmap