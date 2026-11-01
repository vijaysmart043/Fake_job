
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    LimeTextExplainer = None

from src.preprocessing import clean_text, prepare_features
from src.explain import explain_text
from src.url_scanner import scan_job_url
from src.job_categories import get_category_report


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="JobShield - Fake Job Detection",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #2563eb;
    }
    .sub-title {
        color: #64748b;
        font-size: 16px;
    }
    .portal-card {
        border: 1px solid #dbe3ef;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        min-height: 115px;
    }
    .portal-card a {
        text-decoration: none;
        font-weight: 600;
    }
    .small-note {
        color: #64748b;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():
    models = {}

    baseline_path = Path("models/baseline_model.joblib")
    fusion_path = Path("models/fusion_model.joblib")

    if baseline_path.exists():
        models["Baseline AI Model"] = joblib.load(baseline_path)

    if fusion_path.exists():
        models["Advanced Feature Fusion Model"] = joblib.load(fusion_path)

    return models


models = load_models()


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def make_job_record(
    title="",
    company_profile="",
    description="",
    requirements="",
    benefits="",
    employment_type="",
    required_experience="",
    required_education="",
    industry="",
    function="",
    department="",
    location="",
    salary_range="",
    telecommuting=0,
    has_company_logo=0,
    has_questions=0
):
    return {
        "title": title,
        "company_profile": company_profile,
        "description": description,
        "requirements": requirements,
        "benefits": benefits,
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": industry,
        "function": function,
        "department": department,
        "location": location,
        "salary_range": salary_range,
        "telecommuting": telecommuting,
        "has_company_logo": has_company_logo,
        "has_questions": has_questions
    }


def get_job_text(record):
    fields = [
        record.get("title", ""),
        record.get("company_profile", ""),
        record.get("description", ""),
        record.get("requirements", ""),
        record.get("benefits", "")
    ]

    return " ".join(
        clean_text(value)
        for value in fields
        if value
    )


def predict_job(record, model_name):
    model = models[model_name]

    if model_name == "Baseline AI Model":
        text = get_job_text(record)

        prediction = int(model.predict([text])[0])
        probability = float(model.predict_proba([text])[0][1])

        explanation = explain_text(model, text)

    else:
        raw_df = pd.DataFrame([record])
        feature_df = prepare_features(raw_df)

        prediction = int(model.predict(feature_df)[0])
        probability = float(model.predict_proba(feature_df)[0][1])

        explanation = None

    return prediction, probability, explanation


def show_prediction(prediction, probability, explanation=None):
    st.divider()

    st.subheader("Job Screening Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ Potentially Suspicious Job")
            st.write(
                "The AI model has identified patterns that may be associated "
                "with fraudulent job advertisements."
            )
        else:
            st.success("✅ No Strong Fraud Signals Detected")
            st.write(
                "The AI model did not identify strong fraud patterns in the "
                "provided job information."
            )

    with col2:
        st.metric("Model Fraud Score", f"{probability * 100:.2f}%")
        st.progress(max(0.0, min(1.0, probability)))

    st.warning(
        "This is an AI screening result, not proof that a job is genuine or "
        "fraudulent. Verify the employer and offer independently."
    )

    if explanation:
        st.subheader("Why did the AI make this prediction?")

        fraud_signals = explanation.get("fraud_signals", [])
        genuine_signals = explanation.get("genuine_signals", [])

        left, right = st.columns(2)

        with left:
            st.markdown("**Words that pushed the result towards suspicious**")
            if fraud_signals:
                for word, score in fraud_signals:
                    st.write(f"🔴 {word}")
            else:
                st.write("No strong suspicious text signals were found.")

        with right:
            st.markdown("**Words that pushed the result towards genuine**")
            if genuine_signals:
                for word, score in genuine_signals:
                    st.write(f"🟢 {word}")
            else:
                st.write("No strong genuine text signals were found.")


def load_metrics(filename):
    path = Path("reports") / filename

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def run_lime_explanation(model, text, num_features=12, num_samples=1000):
    """Create a local word-level LIME explanation for the baseline model."""
    if LimeTextExplainer is None:
        raise ImportError(
            "LIME is not installed. Activate your venv and run: pip install lime"
        )

    explainer = LimeTextExplainer(
        class_names=["Genuine", "Fraud"],
        random_state=42
    )
    return explainer.explain_instance(
        text_instance=text,
        classifier_fn=model.predict_proba,
        labels=[1],
        num_features=num_features,
        num_samples=num_samples
    )


def load_shap_explanation():
    """Load the saved SHAP result generated by src.explain_shap."""
    path = Path("reports/shap_explanation.json")
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def display_portal_group(title, portals):
    st.subheader(title)

    columns = st.columns(3)

    for index, (name, icon, url, description) in enumerate(portals):
        with columns[index % 3]:
            st.markdown(
                f"""
                <div class="portal-card">
                    <h3>{icon} {name}</h3>
                    <p>{description}</p>
                    <a href="{url}" target="_blank">Visit website ↗</a>
                </div>
                """,
                unsafe_allow_html=True
            )


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🛡️ JobShield</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI-Powered Fake Job Advertisement Detection System</div>',
    unsafe_allow_html=True
)

st.write(
    "Check job advertisements, explore career portals, and understand "
    "how our machine learning model identifies suspicious job postings."
)

if not models:
    st.error(
        "No trained models found. Please run the training scripts first "
        "and check the models folder."
    )
    st.stop()


# --------------------------------------------------
# MAIN NAVIGATION
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Check a Job",
    "🌐 Career Portals",
    "📊 Job Category Explorer",
    "🛡️ Safety Tips",
    "🎓 Project Performance",
    "🧠 Explainability Lab"
])


# ==================================================
# TAB 1 - CHECK A JOB
# ==================================================

with tab1:

    st.header("Check Whether a Job Looks Suspicious")
    st.write(
        "Enter the job information below, or provide the job advertisement "
        "URL to screen it using the trained AI model."
    )

    model_name = st.selectbox(
        "Choose AI Model",
        list(models.keys()),
        help="The baseline model analyses job text. The fusion model also uses structured job details."
    )

    manual_tab, url_tab = st.tabs([
        "📝 Enter Job Details",
        "🔗 Scan Job URL"
    ])

    with manual_tab:

        with st.form("manual_job_form"):

            title = st.text_input("Job Title")
            company_profile = st.text_area("Company Profile")
            description = st.text_area("Job Description")
            requirements = st.text_area("Job Requirements")
            benefits = st.text_area("Benefits Offered")

            st.markdown("#### Additional Job Information")

            c1, c2 = st.columns(2)

            with c1:
                employment_type = st.selectbox(
                    "Employment Type",
                    ["Not provided", "Full-time", "Part-time",
                     "Contract", "Temporary", "Other"]
                )

                required_experience = st.selectbox(
                    "Required Experience",
                    ["Not provided", "Internship", "Entry level",
                     "Associate", "Mid-Senior level", "Executive", "Other"]
                )

                required_education = st.selectbox(
                    "Required Education",
                    ["Not provided", "High School", "Bachelor's Degree",
                     "Master's Degree", "Doctorate", "Other"]
                )

                industry = st.text_input("Industry")

            with c2:
                function = st.text_input("Job Function")
                department = st.text_input("Department")
                location = st.text_input("Job Location")
                salary_range = st.text_input("Salary Range")

            logo = st.selectbox(
                "Does the advertisement show a company logo?",
                ["Not sure", "Yes", "No"]
            )

            questions = st.selectbox(
                "Does the application include screening questions?",
                ["Not sure", "Yes", "No"]
            )

            remote = st.selectbox(
                "Is remote work mentioned?",
                ["Not sure", "Yes", "No"]
            )

            submitted = st.form_submit_button(
                "🔍 Analyse Job Advertisement",
                use_container_width=True
            )

        if submitted:

            if not title.strip() and not description.strip():
                st.warning("Please enter at least a job title or job description.")

            else:
                record = make_job_record(
                    title=title,
                    company_profile=company_profile,
                    description=description,
                    requirements=requirements,
                    benefits=benefits,
                    employment_type=employment_type,
                    required_experience=required_experience,
                    required_education=required_education,
                    industry=industry,
                    function=function,
                    department=department,
                    location=location,
                    salary_range=salary_range,
                    telecommuting=1 if remote == "Yes" else 0,
                    has_company_logo=1 if logo == "Yes" else 0,
                    has_questions=1 if questions == "Yes" else 0
                )

                try:
                    prediction, probability, explanation = predict_job(
                        record, model_name
                    )

                    show_prediction(
                        prediction, probability, explanation
                    )

                except Exception as error:
                    st.error(f"Could not analyse this job: {error}")


    with url_tab:

        st.subheader("Scan a Job Advertisement URL")

        st.write(
            "Paste a publicly accessible job advertisement URL. "
            "The system will try to extract the page text and analyse it."
        )

        job_url = st.text_input(
            "Job Advertisement URL",
            placeholder="https://example.com/job-posting"
        )

        if st.button("🌐 Scan Job URL", use_container_width=True):

            if not job_url.strip():
                st.warning("Please enter a job URL.")

            else:
                try:
                    with st.spinner("Reading job advertisement..."):
                        result = scan_job_url(job_url.strip())

                    if not result:
                        st.error("No readable job information was extracted.")

                    else:
                        page_title = result.get("title", "")
                        meta_description = result.get("meta_description", "")
                        page_text = result.get("page_text", "")

                        if not page_text.strip():
                            st.warning(
                                "The website did not provide readable job text. "
                                "It may require login or JavaScript."
                            )

                        else:
                            st.success("Job page content extracted.")

                            st.write("**Page title:**", page_title or "Not found")

                            with st.expander("View extracted page text"):
                                st.text(page_text[:5000])

                            record = make_job_record(
                                title=page_title,
                                description=page_text,
                                company_profile=meta_description
                            )

                            try:
                                prediction, probability, explanation = predict_job(
                                    record, model_name
                                )

                                show_prediction(
                                    prediction, probability, explanation
                                )

                            except Exception as error:
                                st.error(f"Could not analyse extracted text: {error}")

                except Exception as error:
                    st.error(f"URL scan failed: {error}")

        st.info(
            "URL scanning depends on whether the website allows automated "
            "access. A successful scan does not verify the company's identity "
            "or the authenticity of the job."
        )


# ==================================================
# TAB 2 - CAREER PORTALS
# ==================================================

with tab2:

    st.header("Explore Career and Job Portals")

    st.write(
        "Use these websites to search for jobs. Always verify the individual "
        "job advertisement, even when it appears on a well-known portal."
    )

    portal_group = st.selectbox(
        "Choose Portal Category",
        [
            "Corporate & General Job Portals",
            "MNC Official Career Pages",
            "Startups & Technology Jobs",
            "Freshers & Internships",
            "BPO & Frontline Jobs"
        ]
    )

    portal_data = {

        "Corporate & General Job Portals": [
            ("LinkedIn Jobs", "💼", "https://www.linkedin.com/jobs/",
             "Professional networking and job listings."),
            ("Naukri", "🔎", "https://www.naukri.com/",
             "Jobs across many industries in India."),
            ("Indeed India", "🌍", "https://in.indeed.com/",
             "Search jobs by role and location."),
            ("Foundit", "📌", "https://www.foundit.in/",
             "Job search and career opportunities."),
            ("Glassdoor", "🏢", "https://www.glassdoor.co.in/",
             "Jobs and employer information."),
            ("TimesJobs", "📰", "https://www.timesjobs.com/",
             "Job listings across sectors.")
        ],

        "MNC Official Career Pages": [
            ("TCS Careers", "🏢", "https://www.tcs.com/careers",
             "Official careers page of Tata Consultancy Services."),
            ("Infosys Careers", "💻", "https://www.infosys.com/careers/",
             "Official Infosys job opportunities."),
            ("Wipro Careers", "🌐", "https://careers.wipro.com/",
             "Official Wipro careers portal."),
            ("Accenture Careers", "🚀", "https://www.accenture.com/in-en/careers",
             "Explore Accenture roles and opportunities."),
            ("Amazon Jobs", "📦", "https://www.amazon.jobs/",
             "Official Amazon job search."),
            ("Microsoft Careers", "🪟", "https://careers.microsoft.com/",
             "Official Microsoft careers portal."),
            ("Google Careers", "🔍", "https://www.google.com/about/careers/applications/",
             "Official Google job opportunities.")
        ],

        "Startups & Technology Jobs": [
            ("Wellfound", "🚀", "https://wellfound.com/",
             "Startup jobs and company discovery."),
            ("Cutshort", "⚡", "https://cutshort.io/",
             "Technology and startup opportunities."),
            ("Instahyre", "💡", "https://www.instahyre.com/",
             "Technology and professional hiring."),
            ("Hirist", "👨‍💻", "https://www.hirist.tech/",
             "IT and technology jobs.")
        ],

        "Freshers & Internships": [
            ("Internshala", "🎓", "https://internshala.com/",
             "Internships and entry-level opportunities."),
            ("Freshersworld", "🌱", "https://www.freshersworld.com/",
             "Fresher jobs and recruitment opportunities."),
            ("National Career Service", "🇮🇳", "https://www.ncs.gov.in/",
             "Government-supported career services.")
        ],

        "BPO & Frontline Jobs": [
            ("Apna", "📱", "https://apna.co/",
             "Jobs across frontline and professional roles."),
            ("WorkIndia", "🧑‍💼", "https://www.workindia.in/",
             "Local and frontline job opportunities."),
            ("QuikrJobs", "📋", "https://www.quikr.com/jobs/",
             "Job listings across different categories.")
        ]
    }

    display_portal_group(
        portal_group,
        portal_data[portal_group]
    )

    st.warning(
        "These are examples of career portals, not a complete list of every "
        "MNC or startup. A genuine portal can still contain misleading or "
        "fraudulent individual listings."
    )


# ==================================================
# TAB 3 - JOB CATEGORY EXPLORER
# ==================================================

with tab3:

    st.header("Explore Fraud Patterns by Job Category")

    st.write(
        "See how many genuine and fraudulent advertisements appear in "
        "different job categories in the EMSCAD dataset."
    )

    st.info(
        "These are historical dataset statistics. Categories are assigned "
        "using keyword matching, so the classification is approximate. "
        "The percentages are not current real-world industry fraud rates."
    )

    try:
        report = get_category_report()

        selected_category = st.selectbox(
            "Select a Job Category",
            ["All Categories"] + report["job_category"].tolist()
        )

        if selected_category == "All Categories":
            filtered = report.copy()
        else:
            filtered = report[
                report["job_category"] == selected_category
            ]

        total_jobs = int(filtered["total_jobs"].sum())
        fake_jobs = int(filtered["fake_jobs"].sum())
        genuine_jobs = int(filtered["genuine_jobs"].sum())

        fraud_percentage = (
            fake_jobs / total_jobs * 100 if total_jobs else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Job Posts", f"{total_jobs:,}")
        c2.metric("Flagged as Fraudulent", f"{fake_jobs:,}")
        c3.metric("Genuine Job Posts", f"{genuine_jobs:,}")
        c4.metric("Fraudulent Share", f"{fraud_percentage:.2f}%")

        st.subheader("Fraudulent Share by Category")

        chart_data = report.set_index("job_category")[
            ["fraud_percentage"]
        ].sort_values("fraud_percentage", ascending=True)

        st.bar_chart(chart_data)

        st.subheader("Genuine vs Fraudulent Job Posts")

        count_data = report.set_index("job_category")[
            ["genuine_jobs", "fake_jobs"]
        ]

        st.bar_chart(count_data)

        st.subheader("Category Details")

        display_report = report.copy()
        display_report["fraud_percentage"] = (
            display_report["fraud_percentage"].map(
                lambda value: f"{value:.2f}%"
            )
        )

        display_report = display_report.rename(columns={
            "job_category": "Job Category",
            "total_jobs": "Total Jobs",
            "fake_jobs": "Flagged as Fraudulent",
            "genuine_jobs": "Genuine Jobs",
            "fraud_percentage": "Fraudulent Share"
        })

        st.dataframe(
            display_report,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Note: A high percentage in a small category may be based on "
            "relatively few examples. Always consider both the count and "
            "percentage."
        )

    except Exception as error:
        st.error(f"Could not load category report: {error}")


# ==================================================
# TAB 4 - SAFETY TIPS
# ==================================================

with tab4:

    st.header("How to Stay Safe While Applying for Jobs")

    st.write(
        "Use these basic checks before sharing personal information "
        "or accepting a job offer."
    )

    safety_tips = [
        (
            "💰 Never pay to get a job",
            "Be careful if someone asks for registration fees, interview "
            "fees, security deposits, or payment for guaranteed placement."
        ),
        (
            "🏢 Verify the company",
            "Visit the company's official website and check whether the "
            "job opening is listed on its official careers page."
        ),
        (
            "📧 Check the email address",
            "Look carefully at the sender's email domain. Be cautious of "
            "free email accounts or domains that imitate company names."
        ),
        (
            "📄 Read the job description",
            "Unclear responsibilities, unrealistic salary promises, and "
            "vague company details deserve extra attention."
        ),
        (
            "🔐 Protect personal information",
            "Do not share passwords, OTPs, banking PINs, or sensitive "
            "identity documents with unverified recruiters."
        ),
        (
            "🎥 Be careful with interviews",
            "Verify the recruiter and interview process. Be suspicious "
            "of offers made without a meaningful selection process."
        ),
        (
            "🔗 Check application links",
            "Confirm that application links belong to the employer or "
            "a trusted recruitment platform."
        )
    ]

    for heading, explanation in safety_tips:
        with st.expander(heading):
            st.write(explanation)

    st.success(
        "Remember: An AI prediction is only one screening signal. "
        "Independent verification is always important."
    )


# ==================================================
# TAB 5 - PROJECT PERFORMANCE
# ==================================================

with tab5:

    st.header("How Well Does Our AI Model Perform?")

    st.write(
        "This section explains the model evaluation results in simple "
        "language. The results are from the held-out test dataset."
    )

    baseline_metrics = load_metrics("baseline_metrics.json")
    fusion_metrics = load_metrics("fusion_metrics.json")

    metric_descriptions = {
        "accuracy": (
            "Overall correct predictions",
            "Out of all test jobs, how many did the model classify correctly?"
        ),
        "precision": (
            "Reliability of fraud alerts",
            "When the model flags a job as fraudulent, how often is it correct?"
        ),
        "recall": (
            "Fraud detection coverage",
            "Out of all fraudulent jobs in the test set, how many did the model catch?"
        ),
        "f1_score": (
            "Balance between precision and recall",
            "A combined score that balances catching fraud and avoiding false alarms."
        ),
        "pr_auc": (
            "Fraud detection quality",
            "Measures how well the model balances precision and recall across thresholds."
        ),
        "roc_auc": (
            "Overall class separation",
            "Measures how well the model separates genuine and fraudulent jobs."
        )
    }

    def show_model_performance(metrics, model_title):

        if not metrics:
            st.warning(
                f"Metrics file for {model_title} was not found."
            )
            return

        st.subheader(model_title)

        for key in ["accuracy", "precision", "recall", "f1_score"]:
            if key in metrics:
                label, description = metric_descriptions[key]
                value = float(metrics[key])

                st.markdown(f"**{label}**")
                st.progress(max(0.0, min(1.0, value)))
                st.write(f"{value * 100:.2f}%")
                st.caption(description)

        with st.expander("M.Tech AI/ML Evaluation Details"):

            for key in ["pr_auc", "roc_auc"]:
                if key in metrics:
                    label, description = metric_descriptions[key]
                    st.write(f"**{key.upper()}:** {float(metrics[key]):.4f}")
                    st.caption(description)

            if "confusion_matrix" in metrics:
                st.write("Confusion Matrix")
                st.write(metrics["confusion_matrix"])

            if "classification_report" in metrics:
                st.write("Classification Report")
                st.json(metrics["classification_report"])

    baseline_tab, fusion_tab = st.tabs([
        "Baseline Model",
        "Feature Fusion Model"
    ])

    with baseline_tab:
        show_model_performance(
            baseline_metrics,
            "TF-IDF + Logistic Regression"
        )

    with fusion_tab:
        show_model_performance(
            fusion_metrics,
            "Text + Structured Feature Fusion"
        )

    st.divider()

    st.subheader("What did we learn?")

    st.write(
        "The baseline model achieved a stronger balance of precision, "
        "F1-score and PR-AUC in the current experiment. The feature fusion "
        "model achieved slightly higher fraud recall."
    )

    st.write(
        "This means the fusion model caught a slightly larger share of "
        "fraudulent jobs, but it also produced more false alarms. The best "
        "model depends on whether the priority is catching more suspicious "
        "jobs or reducing incorrect alerts."
    )

    st.warning(
        "These results are based on a random held-out split of the EMSCAD "
        "dataset. The dataset contains older job advertisements, so further "
        "testing on newer job postings is needed before real-world use."
    )


# ==================================================
# TAB 6 - EXPLAINABILITY LAB
# ==================================================

with tab6:

    st.header("Understand Why the Model Made a Prediction")
    st.write(
        "Enter job advertisement text to see a LIME word-level explanation "
        "from the TF-IDF baseline model. The saved SHAP report is shown "
        "separately as a reference explanation from the DistilBERT fusion model."
    )

    st.subheader("1. Live LIME explanation — Baseline model")
    st.caption(
        "LIME perturbs the entered text and estimates which words or phrases "
        "locally push the baseline model towards fraud or genuine."
    )

    with st.form("lime_explainability_form"):
        lime_title = st.text_input("Job Title", key="lime_job_title")
        lime_company = st.text_area("Company Profile", key="lime_company_profile")
        lime_description = st.text_area(
            "Job Description", height=180, key="lime_job_description"
        )
        lime_requirements = st.text_area(
            "Requirements", key="lime_job_requirements"
        )
        lime_benefits = st.text_area("Benefits", key="lime_job_benefits")
        lime_submitted = st.form_submit_button(
            "🧠 Generate LIME Explanation", use_container_width=True
        )

    if lime_submitted:
        lime_record = make_job_record(
            title=lime_title,
            company_profile=lime_company,
            description=lime_description,
            requirements=lime_requirements,
            benefits=lime_benefits
        )
        lime_text = get_job_text(lime_record)

        if not lime_text.strip():
            st.warning("Enter at least some job advertisement text.")
        elif "Baseline AI Model" not in models:
            st.error("The baseline model is not available in the models folder.")
        else:
            try:
                with st.spinner("Generating local LIME explanation..."):
                    lime_model = models["Baseline AI Model"]
                    lime_probabilities = lime_model.predict_proba([lime_text])[0]
                    lime_prediction = int(lime_model.predict([lime_text])[0])
                    lime_explanation = run_lime_explanation(
                        lime_model, lime_text
                    )

                score_col, prediction_col = st.columns(2)
                score_col.metric(
                    "Model Fraud Score", f"{float(lime_probabilities[1]) * 100:.2f}%"
                )
                prediction_col.metric(
                    "Baseline Prediction",
                    "Fraud" if lime_prediction == 1 else "Genuine"
                )

                lime_rows = [
                    {
                        "Word / phrase": word,
                        "Contribution": float(weight),
                        "Direction": "Towards fraud" if weight > 0 else "Towards genuine"
                    }
                    for word, weight in lime_explanation.as_list(label=1)
                ]
                lime_df = pd.DataFrame(lime_rows)
                lime_df["Contribution"] = lime_df["Contribution"].map(
                    lambda value: round(value, 5)
                )
                st.dataframe(lime_df, use_container_width=True, hide_index=True)

                st.caption(
                    "Positive values push this local explanation towards fraud; "
                    "negative values push it towards genuine. LIME is a local "
                    "approximation, not proof or a causal explanation."
                )
            except Exception as error:
                st.error(f"Could not generate the LIME explanation: {error}")

    st.divider()
    st.subheader("2. Saved SHAP explanation — DistilBERT fusion model")
    st.caption(
        "This displays the saved output of python -m src.explain_shap. "
        "It is a reference dataset example and is not calculated from the "
        "text entered in the LIME form above."
    )

    shap_data = load_shap_explanation()
    if not shap_data:
        st.info(
            "No SHAP report found. Run `python -m src.explain_shap` from "
            "the project root to generate reports/shap_explanation.json."
        )
    else:
        shap_col1, shap_col2 = st.columns(2)
        shap_col1.metric("SHAP Example Prediction", str(shap_data.get("prediction", "Not available")))
        shap_score = shap_data.get("fraud_score")
        if isinstance(shap_score, (int, float)):
            shap_col2.metric("SHAP Example Fraud Score", f"{float(shap_score) * 100:.6f}%")
        else:
            shap_col2.metric("SHAP Example Fraud Score", "Not available")

        text_contribution = shap_data.get("text_contribution")
        if text_contribution is None:
            text_contribution = shap_data.get("combined_text_contribution")
        if isinstance(text_contribution, (int, float)):
            st.metric("Aggregated text contribution", f"{float(text_contribution):+.5f}")

        structured = shap_data.get("structured_contributions")
        if structured is None:
            structured = shap_data.get("structured_feature_contributions", {})

        st.markdown("**Structured feature contributions**")
        if isinstance(structured, dict) and structured:
            shap_rows = [
                {"Feature": key, "SHAP contribution": float(value),
                 "Direction": "Towards fraud" if float(value) > 0 else "Towards genuine"}
                for key, value in structured.items()
                if isinstance(value, (int, float))
            ]
        elif isinstance(structured, list):
            shap_rows = []
            for item in structured:
                if isinstance(item, dict):
                    feature_name = item.get("feature", item.get("name", "Unknown"))
                    feature_value = item.get("contribution", item.get("value"))
                    if isinstance(feature_value, (int, float)):
                        shap_rows.append({
                            "Feature": feature_name,
                            "SHAP contribution": float(feature_value),
                            "Direction": "Towards fraud" if float(feature_value) > 0 else "Towards genuine"
                        })
        else:
            shap_rows = []

        if shap_rows:
            st.dataframe(
                pd.DataFrame(shap_rows).sort_values(
                    "SHAP contribution", key=lambda col: col.abs(), ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.json(shap_data)
            st.caption(
                "The saved report's field names differ from the expected "
                "structured-contribution format, so the raw JSON is shown."
            )

        st.warning(
            "SHAP contributions describe how this trained model arrived at "
            "its output for the saved example. They do not establish that a "
            "job is actually fraudulent or genuine."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "JobShield | M.Tech Artificial Intelligence & Machine Learning Project"
)
