# app.py

"""
Streamlit app for AI-based Resume Screening Tool.

Professional features:
- Permission confirmation (HR must confirm they can upload resumes)
- Terms & Privacy section
- Strong no-storage / local-processing messaging
- Optional anonymous metrics logging (no candidate names)
- Explainable score breakdown
- Clean, HR-friendly UI
"""

import os
from datetime import datetime
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    APP_BRAND_EMOJI,
    APP_NAME,
    APP_TAGLINE,
    MODEL_PATH,
    USAGE_METRICS_PATH,
)
from file_parsers import extract_text_from_file
from nlp_utils import build_features


@st.cache_resource
def load_model():
    """Load trained ML model from disk if available, else return None."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def heuristic_score(meta: Dict) -> Dict:
    """
    Fallback scoring when no ML model is available.
    Returns score and component breakdown for explainability.
    """
    similarity = meta["similarity"]
    exp_years = meta["exp_years"]
    common_skills = meta["common_skills"]
    resume_skills = meta["resume_skills"]

    skill_ratio = (
        len(common_skills) / (len(resume_skills) or 1)
        if resume_skills
        else 0.0
    )

    exp_score = min(exp_years / 10.0, 1.0)  # cap at 10+ years

    weight_similarity = 0.5
    weight_skills = 0.3
    weight_experience = 0.2

    score = (
        weight_similarity * similarity
        + weight_skills * skill_ratio
        + weight_experience * exp_score
    )

    breakdown = {
        "similarity_component": weight_similarity * similarity,
        "skills_component": weight_skills * skill_ratio,
        "experience_component": weight_experience * exp_score,
        "similarity_raw": similarity,
        "skill_ratio_raw": skill_ratio,
        "exp_score_raw": exp_score,
    }

    return {"score": float(score), "breakdown": breakdown}


def log_usage(num_resumes: int, model_used: str, scores: List[float]):
    """
    Log anonymous usage metrics (no resume names).
    You can disable this by commenting out this function call in `main()`.
    """
    try:
        os.makedirs(os.path.dirname(USAGE_METRICS_PATH), exist_ok=True)
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "num_resumes": num_resumes,
            "model_used": model_used,
            "avg_score": float(np.mean(scores)) if scores else None,
            "max_score": float(np.max(scores)) if scores else None,
            "min_score": float(np.min(scores)) if scores else None,
        }
        if os.path.exists(USAGE_METRICS_PATH):
            df = pd.read_csv(USAGE_METRICS_PATH)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(USAGE_METRICS_PATH, index=False)
    except Exception:
        # Fail silently; logging is optional
        pass


def render_terms_and_privacy():
    with st.expander("📜 Terms, Privacy & Responsible Use"):
        st.markdown(
            """
**1. Data Handling & Privacy**

- Resumes are processed **in-memory only** during this session.
- This demo app does **not** write uploaded resume contents to disk or any external database.
- You are responsible for running this tool in a secure environment (e.g., HTTPS, restricted access).

**2. Permissions**

- Only upload CVs / resumes if you have **explicit permission** to process them.
- Ensure you comply with your organization’s **data protection** and **privacy** policies.

**3. Limitations**

- This is an **assistive** tool, not an automated hiring decision engine.
- NLP and scoring are **approximate** and may miss or misinterpret some information.
- All outputs must be reviewed by a **human recruiter**.

**4. No Legal Guarantee**

- This project is for educational / experimentation purposes.
- You are responsible for any usage in real hiring workflows.
"""
        )


def main():
    st.set_page_config(
        page_title=APP_NAME,
        layout="wide",
    )

    # Header
    st.title(f"{APP_BRAND_EMOJI} {APP_NAME}")
    st.caption(APP_TAGLINE)

    # Important notes
    st.markdown("### ⚠️ Important Notes Before Uploading Resumes")
    st.info(
        """
- 🧴 **No Permanent Storage**: Resumes are processed **only in memory** for this session.  
- 🔍 **Ranking Only, No Auto-Rejection**: The tool **scores and ranks** candidates; it does **not** automatically reject anyone.  
- 📝 **Heuristic + ML Limitations**: Skill detection and experience extraction are automated and may not be 100% accurate.  
- 🎯 **Job Description Required**: Without a properly written JD, scores are **not meaningful**.  
"""
    )

    render_terms_and_privacy()

    st.markdown("---")

    # Sidebar – Job Description & controls
    st.sidebar.header("Job Description & Settings")

    jd_text = st.sidebar.text_area(
        "Paste the Job Description here",
        height=250,
        help="This text will be used to evaluate how well each resume matches the role.",
    )

    permission_confirmed = st.sidebar.checkbox(
        "I confirm I have permission to upload and process these resumes.",
        value=False,
        help="You must have the right to process candidate data."
    )

    enable_logging = st.sidebar.checkbox(
        "Enable anonymous usage metrics logging",
        value=True,
        help="Logs only aggregated stats (no candidate names or resume text)."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "_Tip: Train your own model with `train_model.py` for role-specific scoring._"
    )

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload candidate resumes (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Files are processed temporarily in memory and never stored by this app."
    )

    col_run, col_clear = st.columns([1, 1])
    with col_run:
        run_button = st.button("Run Screening ✅")
    with col_clear:
        clear_button = st.button("Clear Results 🧹")

    # Clear button just re-renders page; Streamlit will forget previous results
    if clear_button:
        st.experimental_rerun()

    model = load_model()
    if model is None:
        st.warning(
            "No trained ML model found at `models/resume_ranker.pkl`. "
            "The app will use a heuristic scoring method instead.\n\n"
            "Train a model by running `python train_model.py` after preparing "
            "`data/training_jd.txt` and `data/training_data.csv`."
        )

    if run_button:
        # Pre-checks
        if not permission_confirmed:
            st.error(
                "You must confirm that you have permission to upload and process these resumes."
            )
            return

        if not jd_text.strip():
            st.error(
                "Please paste a Job Description in the sidebar first. "
                "The scoring is meaningless without a JD."
            )
            return

        if not uploaded_files:
            st.error("Please upload at least one resume file.")
            return

        if len(uploaded_files) == 1:
            st.warning(
                "You uploaded only **one** resume. You will still get a score, "
                "but ranking is useful **only** when multiple resumes are uploaded."
            )

        results = []

        with st.spinner("Processing resumes..."):
            for file in uploaded_files:
                try:
                    resume_text = extract_text_from_file(file)
                except Exception as e:
                    st.warning(f"Error reading {file.name}: {e}")
                    continue

                features, meta = build_features(jd_text, resume_text)

                if model is not None:
                    prob = model.predict_proba(features.reshape(1, -1))[:, 1]
                    score = float(prob[0])
                    model_used = "ML Model (Logistic Regression)"
                    breakdown = {
                        "similarity_raw": meta["similarity"],
                        "skill_ratio_raw": (
                            len(meta["common_skills"])
                            / (len(meta["resume_skills"]) or 1)
                            if meta["resume_skills"] else 0.0
                        ),
                        "exp_score_raw": min(meta["exp_years"] / 10.0, 1.0),
                    }
                else:
                    heuristic = heuristic_score(meta)
                    score = heuristic["score"]
                    model_used = "Heuristic Scoring"
                    breakdown = heuristic["breakdown"]

                results.append(
                    {
                        "file_name": file.name,
                        "score": score,
                        "model_used": model_used,
                        "breakdown": breakdown,
                        **meta,
                    }
                )

        if not results:
            st.warning("No resumes could be processed.")
            return

        # Anonymous usage logging (no names)
        if enable_logging:
            log_usage(
                num_resumes=len(results),
                model_used="ML" if model is not None else "Heuristic",
                scores=[r["score"] for r in results],
            )

        # Sort candidates by score descending
        results = sorted(results, key=lambda r: r["score"], reverse=True)

        st.subheader("Ranked Candidates")

        for idx, res in enumerate(results, start=1):
            st.markdown(f"### {idx}. {res['file_name']}")
            st.write(f"**Score:** `{res['score']:.3f}`")
            st.write(f"**Similarity (JD–Resume):** `{res['similarity']:.3f}`")
            st.write(f"**Estimated Experience:** `{res['exp_years']:.1f}` years")
            st.write(f"**Scoring Method:** {res['model_used']}")

            # Explainability: score breakdown
            with st.expander("📊 Score Breakdown & Matched Skills"):
                st.markdown("**Score Components (normalized):**")
                # Use simple text-based representation to avoid extra libs
                b = res["breakdown"]
                st.write(
                    f"- Similarity component: `{b.get('similarity_component', 0):.3f}`"
                )
                st.write(
                    f"- Skills component: `{b.get('skills_component', 0):.3f}`"
                )
                st.write(
                    f"- Experience component: `{b.get('experience_component', 0):.3f}`"
                )

                st.markdown("**Raw feature values:**")
                st.write(
                    f"- JD–Resume similarity: `{b.get('similarity_raw', res['similarity']):.3f}`"
                )
                st.write(
                    f"- Skill match ratio: `{b.get('skill_ratio_raw', 0):.3f}`"
                )
                st.write(
                    f"- Normalized experience score (0–1): `{b.get('exp_score_raw', 0):.3f}`"
                )

                st.markdown("**JD Skills:**")
                st.write(", ".join(res["jd_skills"]) or "_None detected_")

                st.markdown("**Resume Skills:**")
                st.write(", ".join(res["resume_skills"]) or "_None detected_")

                st.markdown("**Matched Skills:**")
                st.write(", ".join(res["common_skills"]) or "_No matches_")

            st.markdown("---")

    # Footer disclaimer
    st.markdown(
        """
---
🛡 **Disclaimer**  
This is an AI-based assistance tool. All scoring results should be reviewed by a human recruiter before making hiring decisions.
"""
    )


if __name__ == "__main__":
    main()
