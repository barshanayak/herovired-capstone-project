import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.matching_engine import ResumeJDMatcher
from src.preprocessing import clean_text, read_pdf_resume


st.set_page_config(
    page_title="Resume JD Matching System",
    page_icon="📄",
    layout="wide"
)


@st.cache_resource
def load_matcher():
    artifact_dir = PROJECT_ROOT / "models"
    return ResumeJDMatcher(artifact_dir=artifact_dir)

matcher = load_matcher()
def render_skill_list(title, skills, status_type="info"):
    st.subheader(title)

    text = ", ".join(skills) if skills else "No skills found"

    if status_type == "success":
        st.success(text)
    elif status_type == "warning":
        st.warning(text)
    else:
        st.info(text)


st.title("📄 Resume Screening and Job Description Matching System")

st.markdown(
    """
This HR Tech prototype ranks multiple PDF resumes against a job description using:

- **Skill-gap analysis**
- **Sentence-BERT semantic similarity**
- **Candidate ranking**
- **Recruiter-friendly recommendation**
"""
)

st.sidebar.header("Project Information")
st.sidebar.info(
    """
**AI/ML Capstone Project**

Domain: HR Tech / NLP  
Model: SBERT + Skill Matching  
Prototype: Streamlit Dashboard
"""
)

st.sidebar.header("Scoring Logic")
st.sidebar.markdown(
    """
**Final Match Score**

`60% Skill Score + 40% Semantic Score`
"""
)

st.header("1. Enter Job Description")

jd_text = st.text_area(
    "Paste the Job Description",
    height=260,
    placeholder="Paste the complete job description here..."
)

st.header("2. Upload Candidate Resumes")

uploaded_files = st.file_uploader(
    "Upload one or more resume PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

results_df = st.session_state.get("results_df")

analyze_button = st.button("Analyze Resumes", type="primary")

if analyze_button:

    if not jd_text.strip():
        st.error("Please enter a job description.")
        st.session_state["results_df"] = None

    elif not uploaded_files:
        st.error("Please upload at least one PDF resume.")
        st.session_state["results_df"] = None

    else:
        jd_text = clean_text(jd_text)
        results = []

        with st.spinner("Analyzing resumes... Please wait."):
            for uploaded_file in uploaded_files:
                resume_text = read_pdf_resume(uploaded_file)

                if not resume_text:
                    continue

                result = matcher.predict_candidate(jd_text, resume_text)

                results.append({
                    "Rank": None,
                    "Candidate": uploaded_file.name,
                    "Skill Score (%)": result["skill_score"],
                    "Semantic Score (%)": result["semantic_score"],
                    "Final Match Score (%)": result["final_score"],
                    "Fit Category": result["fit_category"],
                    "Recommendation": result["recommendation"],
                    "Matched Skills": ", ".join(result["matched_skills"]),
                    "Missing Skills": ", ".join(result["missing_skills"]),
                    "JD Skills": ", ".join(result["jd_skills"]),
                    "Resume Skills": ", ".join(result["resume_skills"]),
                })

        if not results:
            st.error("No readable resume content found. Please upload valid text-based PDF resume files.")
            st.session_state["results_df"] = None

        else:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(
                by="Final Match Score (%)",
                ascending=False
            ).reset_index(drop=True)
            results_df["Rank"] = range(1, len(results_df) + 1)

            st.session_state["results_df"] = results_df

results_df = st.session_state.get("results_df")

if results_df is not None:
    display_df = results_df[
        [
            "Rank",
            "Candidate",
            "Skill Score (%)",
            "Semantic Score (%)",
            "Final Match Score (%)",
            "Fit Category",
            "Recommendation",
        ]
    ]

    st.header("3. Candidate Ranking Results")
    st.dataframe(display_df, width="stretch")

    st.subheader("Top 3 Candidates")
    st.dataframe(display_df.head(3), width="stretch")

    fig = px.bar(
        display_df,
        x="Candidate",
        y="Final Match Score (%)",
        color="Fit Category",
        title="Candidate Ranking by Final Match Score",
        text="Final Match Score (%)"
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Candidate",
        yaxis_title="Final Match Score (%)"
    )

    st.plotly_chart(fig, width="stretch")
    top_candidate = results_df.iloc[0]

    st.header("4. Top Candidate Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Top Candidate", top_candidate["Candidate"])

    with col2:
        st.metric("Skill Score", f'{top_candidate["Skill Score (%)"]}%')

    with col3:
        st.metric("Semantic Score", f'{top_candidate["Semantic Score (%)"]}%')

    with col4:
        st.metric("Final Score", f'{top_candidate["Final Match Score (%)"]}%')

    st.subheader("Fit Category")
    st.info(top_candidate["Fit Category"])

    st.subheader("Recommendation")
    st.success(top_candidate["Recommendation"])

    render_skill_list(
        "Matched Skills",
        top_candidate["Matched Skills"].split(", ") if top_candidate["Matched Skills"] else [],
        "success"
    )

    render_skill_list(
        "Missing Skills",
        top_candidate["Missing Skills"].split(", ") if top_candidate["Missing Skills"] else [],
        "warning"
    )

    st.header("5. Detailed Candidate Skill Gap Analysis")

    selected_candidate = st.selectbox(
        "Select candidate to inspect",
        results_df["Candidate"].tolist()
    )

    candidate_row = results_df[
        results_df["Candidate"] == selected_candidate
    ].iloc[0]

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Skill Score", f'{candidate_row["Skill Score (%)"]}%')

    with col_b:
        st.metric("Semantic Score", f'{candidate_row["Semantic Score (%)"]}%')

    with col_c:
        st.metric("Final Score", f'{candidate_row["Final Match Score (%)"]}%')

    st.subheader("Fit Category")
    st.info(candidate_row["Fit Category"])

    st.subheader("Recommendation")
    st.success(candidate_row["Recommendation"])

    render_skill_list(
        "Job Description Skills",
        candidate_row["JD Skills"].split(", ") if candidate_row["JD Skills"] else [],
        "info"
    )

    render_skill_list(
        "Candidate Resume Skills",
        candidate_row["Resume Skills"].split(", ") if candidate_row["Resume Skills"] else [],
        "info"
    )

    render_skill_list(
        "Matched Skills",
        candidate_row["Matched Skills"].split(", ") if candidate_row["Matched Skills"] else [],
        "success"
    )

    render_skill_list(
        "Missing Skills",
        candidate_row["Missing Skills"].split(", ") if candidate_row["Missing Skills"] else [],
        "warning"
    )

    st.header("6. Download Results")

    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Complete Ranking Results as CSV",
        data=csv,
        file_name="resume_jd_matching_results.csv",
        mime="text/csv"
    )