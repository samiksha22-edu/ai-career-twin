import streamlit as st
import spacy
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules.resume_parser import ResumeParser
from modules.skill_extractor import SkillExtractor
from modules.career_predictor import CareerPredictor
from modules.job_matcher import JobMatcher
from modules.skill_gap import SkillGapAnalyzer
from modules.roadmap import RoadmapGenerator


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Career Twin System",
    page_icon="🎯",
    layout="wide"
)


# --------------------------------------------------
# LOAD NLP MODEL
# --------------------------------------------------

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")


nlp = load_nlp()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🛠 Controls & Options")

target_role_selected = st.sidebar.selectbox(
    "Select Target Career Role",
    [
        "Data Analyst",
        "Data Scientist",
        "ML Engineer",
        "Software Developer",
        "Business Analyst",
        "AI Engineer",
        "Backend Developer",
        "Frontend Developer",
        "Cloud Engineer",
        "DevOps Engineer",
        "Cybersecurity Analyst",
        "Database Administrator",
        "BI Analyst",
        "NLP Engineer"
    ]
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎯 AI-Based Career Twin & Skill Gap Prediction System")

st.markdown(
    "Upload your resume to extract skills, predict career paths, "
    "analyze skill gaps, and generate your career twin profile."
)


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Resume (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"]
)


# --------------------------------------------------
# RESUME ANALYSIS
# --------------------------------------------------

if uploaded_file is not None:

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 Analyze Profile"):

        with st.spinner(
            "Extracting text and running NLP algorithms..."
        ):

            # 1. Resume Parsing
            parser = ResumeParser(file_path)
            raw_text = parser.extract_text()

            # 2. Skill Extraction
            extractor = SkillExtractor()
            extracted_skills = extractor.extract_skills_semantic(
                raw_text
            )

            # 3. Career Prediction
            predictor = CareerPredictor()
            predictor.load_pipeline()

            prediction_res = predictor.predict_career(
                extracted_skills
            )

            # 4. Job Matching
            matcher = JobMatcher()

            match_scores = matcher.match_jobs(
                extracted_skills
            )

            # 5. Skill Gap Analysis
            gap_analyzer = SkillGapAnalyzer()

            gap_res = gap_analyzer.analyze_gap(
                extracted_skills,
                target_role_selected
            )

            # 6. Roadmap Generation
            roadmap_gen = RoadmapGenerator()

            roadmap_res = roadmap_gen.generate_roadmap(
                gap_res.get("missing_skills", [])
            )

            # Store results
            st.session_state.raw_text = raw_text
            st.session_state.extracted_skills = extracted_skills
            st.session_state.prediction_res = prediction_res
            st.session_state.match_scores = match_scores
            st.session_state.gap_res = gap_res
            st.session_state.roadmap_res = roadmap_res

            st.session_state.analyzed = True


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if st.session_state.analyzed:

    st.success("Analysis Complete!")

    # Top Metrics
    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Predicted Role",
        st.session_state.prediction_res["predicted_role"]
    )

    m2.metric(
        "Target Role Match",
        f"{st.session_state.gap_res['readiness_score']}%"
    )

    m3.metric(
        "Skills Detected",
        len(st.session_state.extracted_skills)
    )

    m4.metric(
        "Skill Gaps",
        len(st.session_state.gap_res["missing_skills"])
    )

    st.markdown("---")


    # --------------------------------------------------
    # TABS
    # --------------------------------------------------

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "👤 Career Twin",
            "🎯 Skill Gap",
            "💼 Job Matching",
            "🗺 Roadmap",
            "📈 Model Benchmarks"
        ]
    )


    # --------------------------------------------------
    # TAB 1 - CAREER TWIN
    # --------------------------------------------------

    with tab1:

        st.subheader("Your AI Career Twin")

        col_left, col_right = st.columns(2)

        with col_left:

            st.info(
                f"**Primary Career Classification:** "
                f"{st.session_state.prediction_res['predicted_role']}"
            )

            st.write(
                f"**Model Confidence:** "
                f"{round(st.session_state.prediction_res['confidence'] * 100, 2)}%"
            )

            st.write("**Extracted Skill Set:**")

            st.write(
                ", ".join(
                    [
                        f"`{s}`"
                        for s in st.session_state.extracted_skills
                    ]
                )
            )


        with col_right:

            st.subheader("Explainable AI Insights")

            st.write(
                f"Why predicted as "
                f"**{st.session_state.prediction_res['predicted_role']}**?"
            )

            for skill in st.session_state.extracted_skills[:5]:

                st.write(
                    f"✓ Strong presence of **{skill}** "
                    f"matches role patterns."
                )


    # --------------------------------------------------
    # TAB 2 - SKILL GAP
    # --------------------------------------------------

    with tab2:

        st.subheader(
            f"Skill Gap Analysis for Target Role: "
            f"{target_role_selected}"
        )

        c1, c2 = st.columns(2)


        with c1:

            st.write("🟢 **Matched / Possessed Skills:**")

            matched = st.session_state.gap_res["matched_skills"]

            if matched:

                for skill in matched:

                    st.success(
                        f"✓ {skill}"
                    )

            else:

                st.write(
                    "No direct matches found."
                )


        with c2:

            st.write("🔴 **Missing Skills (Prioritized):**")

            missing = st.session_state.gap_res["missing_skills"]

            if missing:

                for item in missing:

                    if item["priority"] == "High":
                        color = "red"

                    elif item["priority"] == "Medium":
                        color = "orange"

                    else:
                        color = "green"

                    st.markdown(
                        f":{color}[● **{item['skill']}** "
                        f"({item['priority']} Priority)]"
                    )

            else:

                st.write(
                    "No skill gaps detected for this role!"
                )


    # --------------------------------------------------
    # TAB 3 - JOB MATCHING
    # --------------------------------------------------

    with tab3:

        st.subheader(
            "Job Profile Match Analysis"
        )

        df_match = pd.DataFrame(
            st.session_state.match_scores
        )

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        sns.barplot(
            data=df_match.head(8),
            x="match_score",
            y="job_role",
            ax=ax
        )

        ax.set_xlabel(
            "Match Score (%)"
        )

        ax.set_ylabel(
            "Job Role"
        )

        st.pyplot(fig)


    # --------------------------------------------------
    # TAB 4 - LEARNING ROADMAP
    # --------------------------------------------------

    with tab4:

        st.subheader(
            "Personalized Learning Pathway"
        )

        for step in st.session_state.roadmap_res:

            with st.expander(
                f"📌 {step['month']} — {step['focus']}"
            ):

                st.write(
                    f"**Recommended Target Level:** "
                    f"{step['level']}"
                )


    # --------------------------------------------------
    # TAB 5 - ML BENCHMARKS
    # --------------------------------------------------

    with tab5:

        st.subheader(
            "Machine Learning Algorithm Comparison"
        )

        if st.button(
            "Run ML Evaluation Benchmarks"
        ):

            with st.spinner(
                "Evaluating models..."
            ):

                predictor = CareerPredictor()

                benchmarks = (
                    predictor.evaluate_multiple_models()
                )

            st.dataframe(
                benchmarks,
                hide_index=True
            )