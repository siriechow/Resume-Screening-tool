# nlp_utils.py

"""
NLP utilities using spaCy for:
- Loading model
- Skill extraction with PhraseMatcher
- JD–Resume similarity
- Heuristic years-of-experience estimation
- Feature vector creation for the ML model
"""

import re
from typing import Dict, List, Tuple

import numpy as np
import spacy
from spacy.matcher import PhraseMatcher

from config import SKILLS, SPACY_MODEL_NAME

_nlp = None
_skill_matcher = None


def get_nlp():
    """Lazy-load and cache the spaCy NLP model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(SPACY_MODEL_NAME)
    return _nlp


def get_skill_matcher():
    """Create and cache a PhraseMatcher for skills."""
    global _skill_matcher
    if _skill_matcher is None:
        nlp = get_nlp()
        patterns = [nlp.make_doc(skill) for skill in SKILLS]
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        matcher.add("SKILLS", patterns)
        _skill_matcher = matcher
    return _skill_matcher


def extract_skills(text: str) -> List[str]:
    """Extract skills from text using PhraseMatcher over a predefined skill list."""
    nlp = get_nlp()
    matcher = get_skill_matcher()
    doc = nlp(text)
    matches = matcher(doc)

    found = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        found.add(span.text.lower())

    return sorted(list(found))


def estimate_years_of_experience(text: str) -> float:
    """
    Very rough heuristic to estimate years of experience from resume text.
    Looks for patterns like: 'X years', 'X+ years'.
    """
    pattern = r"(\d+)\s*\+?\s*years"
    matches = re.findall(pattern, text.lower())
    years = [int(m) for m in matches]

    if years:
        return float(max(years))
    return 0.0


def get_jd_resume_similarity(jd_text: str, resume_text: str) -> float:
    """
    Compute semantic similarity between Job Description and Resume using spaCy vectors.
    If vectors are not available (empty), fall back to 0.0 and avoid warnings.
    """
    nlp = get_nlp()
    jd_doc = nlp(jd_text)
    resume_doc = nlp(resume_text)

    # If vectors are empty, avoid unreliable similarity
    if jd_doc.vector_norm == 0 or resume_doc.vector_norm == 0:
        # You could also fall back to simple keyword overlap instead of 0.0.
        return 0.0

    return float(jd_doc.similarity(resume_doc))



def build_features(
    jd_text: str,
    resume_text: str,
) -> Tuple[np.ndarray, Dict]:
    """
    Build a numerical feature vector for a given (JD, resume) pair.

    Features:
    - num_matching_skills
    - skill_match_ratio
    - jd_resume_similarity
    - estimated_years_of_experience
    - has_minimum_skill_match (binary)

    Returns:
        features: np.ndarray of shape (5,)
        meta: dict with extracted info for UI (matched skills, etc.)
    """
    resume_text_lower = resume_text.lower()
    jd_text_lower = jd_text.lower()

    jd_skills = extract_skills(jd_text_lower)
    resume_skills = extract_skills(resume_text_lower)

    jd_skill_set = set(jd_skills)
    resume_skill_set = set(resume_skills)
    common_skills = jd_skill_set & resume_skill_set

    num_matching_skills = len(common_skills)
    num_jd_skills = len(jd_skill_set) or 1
    skill_match_ratio = num_matching_skills / num_jd_skills

    similarity = get_jd_resume_similarity(jd_text, resume_text)
    exp_years = estimate_years_of_experience(resume_text)

    has_minimum_skills = 1 if num_matching_skills >= 3 else 0

    features = np.array(
        [
            num_matching_skills,
            skill_match_ratio,
            similarity,
            exp_years,
            has_minimum_skills,
        ],
        dtype=float,
    )

    meta = {
        "jd_skills": sorted(list(jd_skill_set)),
        "resume_skills": sorted(list(resume_skill_set)),
        "common_skills": sorted(list(common_skills)),
        "similarity": similarity,
        "exp_years": exp_years,
    }

    return features, meta
