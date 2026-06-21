import re
from src.preprocessing import clean_text


DEFAULT_SKILLS = [
    "python", "java", "sql", "aws", "azure", "gcp", "docker", "kubernetes",
    "spark", "hadoop", "tableau", "power bi", "excel", "machine learning",
    "deep learning", "tensorflow", "pytorch", "etl", "airflow", "snowflake",
    "linux", "git", "jira", "business analyst", "risk management",
    "accounting", "finance", "servicenow", "data analysis", "data engineering",
    "dbt", "redshift", "mysql", "postgresql", "oracle", "nosql",
    "power apps", "data warehousing", "data modeling", "api", "agile",
    "scrum", "itil", "incident management", "change management",
    "problem management", "service catalog", "emr", "rds", "sql server",
    "teradata", "sas", "vba", "javascript", "react", "node", "fastapi",
    "django", "flask", "rest api", "microservices", "ci/cd", "jenkins",
    "terraform", "mongodb", "redis"
]


def extract_skills(text: str, skills=None):
    skills = skills or DEFAULT_SKILLS
    text = clean_text(text).lower()

    found_skills = []

    for skill in skills:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text):
            found_skills.append(skill)

    return sorted(list(set(found_skills)))


def calculate_skill_gap(jd_text: str, resume_text: str, skills=None):
    skills = skills or DEFAULT_SKILLS

    jd_skills = extract_skills(jd_text, skills)
    resume_skills = extract_skills(resume_text, skills)

    matched_skills = sorted(list(set(jd_skills).intersection(set(resume_skills))))
    missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))

    if len(jd_skills) == 0:
        skill_score = 0.0
    else:
        skill_score = round((len(matched_skills) / len(jd_skills)) * 100, 2)

    return {
        "jd_skills": jd_skills,
        "resume_skills": resume_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_score": skill_score,
    }