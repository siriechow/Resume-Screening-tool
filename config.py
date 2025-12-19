# config.py

"""
Configuration file for the Resume Screening Tool.
Customize skills, paths, spaCy model, and logging here.
"""

# Path where the trained ML model will be stored/loaded
MODEL_PATH = "models/resume_ranker.pkl"

# Optional anonymous metrics log (no candidate names stored)
USAGE_METRICS_PATH = "data/usage_metrics.csv"

# spaCy model name (make sure to download it before use)
# Recommended: en_core_web_lg (has word vectors for similarity)
SPACY_MODEL_NAME = "en_core_web_lg"

# Basic skills list (you can extend this as needed)
SKILLS = [
    # Programming Languages
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "go", "rust",

    # Web
    "html", "css", "react", "angular", "vue", "node.js", "django",
    "flask", "fastapi",

    # Data / ML
    "machine learning", "deep learning", "data science",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "nlp", "natural language processing",

    # Databases
    "mysql", "postgresql", "mongodb", "oracle", "redis",

    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd",

    # Tools / Misc
    "git", "linux", "rest api", "microservices",
]

# Branding options (just for text/icons in the UI)
APP_NAME = "AI Resume Screening Tool"
APP_TAGLINE = "Enterprise-style, privacy-conscious candidate ranking."
APP_BRAND_EMOJI = "🧠"
