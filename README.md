# Resume-Screening-tool
An end-to-end AI Resume Screening Tool built with spaCy, scikit-learn, and Streamlit. It extracts skills, estimates experience, computes semantic similarity between job descriptions and resumes, and ranks candidates using an ML model or heuristic scoring — all with privacy-first, in-memory processing and explainable results.
# 🧠 AI Resume Screening Tool  
### spaCy · Machine Learning · Streamlit

An end-to-end **AI-powered Resume Screening Tool** that uses **Natural Language Processing (NLP)** and **Machine Learning** to rank resumes against a given **Job Description (JD)**.  
Built with a **privacy-first, human-in-the-loop** approach and designed to resemble a real HR product rather than a simple demo.

---

## 🚀 Overview

Recruiters often receive hundreds of resumes for a single job opening.  
This project demonstrates how **AI can assist (not replace)** human decision-making by:

- Extracting relevant skills from resumes
- Estimating years of experience
- Measuring semantic similarity between resumes and job descriptions
- Ranking candidates based on relevance

All processing is done **locally and in memory**, with **no permanent resume storage**.

---

## ✨ Key Features

- 📄 Upload multiple resumes (**PDF / DOCX / TXT**)
- 📝 Paste a **Job Description**
- 🧠 NLP using **spaCy**
  - Skill extraction
  - Semantic similarity (JD ↔ Resume)
  - Experience estimation
- 📊 Candidate scoring using:
  - **Machine Learning model** (Logistic Regression), or
  - **Heuristic scoring** (fallback)
- 🔍 Explainable results (score breakdown)
- 🛡 Privacy-first design (no resume storage)
- ✅ Permission confirmation before processing
- 📜 Terms & responsible-use disclaimer
- 📈 Optional anonymous usage metrics (no PII)

---

## 🛠 Tech Stack

| Component | Technology |
|---------|------------|
| Programming Language | Python 3.8+ |
| NLP | spaCy (`en_core_web_lg`) |
| Machine Learning | scikit-learn |
| Web UI | Streamlit |
| File Parsing | pdfminer.six, python-docx |
| Utilities | NumPy, Pandas, Joblib |

---

## 📂 Project Structure

```text
resume-screening-tool/
│
├── app.py                 # Streamlit UI + application logic
├── train_model.py         # ML training script
├── nlp_utils.py           # NLP utilities (spaCy, feature extraction)
├── file_parsers.py        # Resume file readers (PDF, DOCX, TXT)
├── config.py              # Configuration (skills, paths, branding)
├── requirements.txt
├── README.md
│
├── data/
│   ├── training_data.csv  # Labeled data for ML training
│   └── training_jd.txt    # Job Description used during training
│
└── models/
    └── resume_ranker.pkl  # Trained ML model (generated)
### 🔑 Prerequisites

Ensure you have the following installed on your system:

- Git
- Python (as required by the project)
- Package manager (`pip`)
- Basic knowledge of command-line usage
```
---

## 📥 Installation
### Clone the repository
```bash
git clone https://github.com/siriechow/Resume-Screening-tool
```
### Navigate into the project directory
```bash
cd Resume-Screening-tool
```
### Install dependencies
```bash
pip install -r requirements.txt   # for Python projects
```
