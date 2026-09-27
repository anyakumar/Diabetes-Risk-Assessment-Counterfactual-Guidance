import io
import os
import sys
import uuid
import datetime
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Safe Windows console encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Page configuration
st.set_page_config(
    page_title="DiaGuard — Clinical Diabetes Risk & Intervention Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Medical SaaS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Clean Top Header */
    .app-header {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.875rem;
        color: #64748B;
        margin-top: 0.25rem;
        margin-bottom: 0;
    }
    
    /* Structured Clinical Card */
    .clinical-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    .card-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1E293B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.85rem;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 0.5rem;
    }
    
    /* Clinical Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .badge-teal { background-color: #CCFBF1; color: #0F766E; }
    .badge-slate { background-color: #F1F5F9; color: #475569; }
    
    /* Risk Tiers */
    .tier-low {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #15803D;
        padding: 1rem;
        border-radius: 6px;
    }
    .tier-moderate {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #B45309;
        padding: 1rem;
        border-radius: 6px;
    }
    .tier-high {
        background-color: #FEF2F2;
        border: 1px solid #FECDD3;
        color: #B91C1C;
        padding: 1rem;
        border-radius: 6px;
    }
    
    .score-display {
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1;
        margin: 0.3rem 0;
    }
    
    /* Preset button row */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.15s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Load Model Bundle
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "diaguard_model.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Run `python src/train.py` first.")
        st.stop()
    return joblib.load(model_path)

bundle = load_model()
model = bundle["model"]
feature_names = bundle["feature_names"]
metrics = bundle.get("metrics", {})
feature_importances = bundle.get("feature_importances", {})

# Standard Age & Health Categorization (CDC BRFSS codebook)
AGE_MAP = {
    "18 – 24 years": 1, "25 – 29 years": 2, "30 – 34 years": 3, "35 – 39 years": 4,
    "40 – 44 years": 5, "45 – 49 years": 6, "50 – 54 years": 7, "55 – 59 years": 8,
    "60 – 64 years": 9, "65 – 69 years": 10, "70 – 74 years": 11, "75 – 79 years": 12,
    "80+ years": 13
}

GEN_HLTH_MAP = {
    "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
}

# Sidebar: System Telemetry & Model Versioning
with st.sidebar:
    st.markdown("### System Information")
    st.markdown("""
    <div style="font-size: 0.85rem; color: #475569; line-height: 1.6;">
        <strong>Platform:</strong> DiaGuard CDS Engine<br>
        <strong>Model:</strong> Calibrated Gradient Boosting (XGBoost)<br>
        <strong>Training Cohort:</strong> CDC BRFSS (n = 253,680)<br>
        <strong>Validation ROC-AUC:</strong> 0.8272<br>
        <strong>Clinical Sensitivity:</strong> 79.59%<br>
        <strong>Serving Interface:</strong> REST API & Clinical Portal
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### Quick Clinical Presets")
    st.caption("Load verified representative patient records:")
    preset_choice = st.radio(
        "Select Patient Profile",
        ["Default Intake", "High-Risk Metabolic", "Borderline Pre-diabetic", "Active Low-Risk"],
        label_visibility="collapsed"
    )

# Setup initial values based on preset
if preset_choice == "High-Risk Metabolic":
    init_age = "55 – 59 years"
    init_sex = "Male"
    init_bp = True
    init_chol = True
    init_bmi = 34.2
    init_phys = False
    init_smoke = True
    init_hlth = "Fair"
    init_diff = True
elif preset_choice == "Borderline Pre-diabetic":
    init_age = "45 – 49 years"
    init_sex = "Female"
    init_bp = True
    init_chol = False
    init_bmi = 28.6
    init_phys = True
    init_smoke = False
    init_hlth = "Good"
    init_diff = False
elif preset_choice == "Active Low-Risk":
    init_age = "30 – 34 years"
    init_sex = "Female"
    init_bp = False
    init_chol = False
    init_bmi = 22.4
    init_phys = True
    init_smoke = False
    init_hlth = "Excellent"
    init_diff = False
else:
    init_age = "50 – 54 years"
    init_sex = "Male"
    init_bp = True
    init_chol = True
    init_bmi = 29.5
    init_phys = False
    init_smoke = True
    init_hlth = "Good"
    init_diff = False

# Application Banner
current_date = datetime.date.today().strftime("%B %d, %Y")
st.markdown(f"""
<div class="app-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="app-title">DiaGuard Clinical Decision Support</h1>
            <p class="app-subtitle">Population-health predictive screening and counterfactual lifestyle intervention analysis.</p>
        </div>
        <div style="text-align: right;">
            <span class="badge badge-teal">Production Model v1.0</span>
            <span class="badge badge-slate" style="margin-left: 0.35rem;">Session: {current_date}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation
tabs = st.tabs([
    "Patient Triage & Risk Assessment",
    "Counterfactual Lifestyle Simulation",
    "Model Explainability & Clinical Drivers",
    "Batch Cohort Screening",
    "EHR Consultation Note"
])

# ================= TAB 1: PATIENT TRIAGE =================
with tabs[0]:
    col_input, col_results = st.columns([1.6, 1.4], gap="large")
    
    with col_input:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">1. Patient Demographics & Vitals</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            age_label = st.selectbox("Age Range", list(AGE_MAP.keys()), index=list(AGE_MAP.keys()).index(init_age))
            age_val = AGE_MAP[age_label]
            sex_label = st.radio("Biological Sex", ["Female", "Male"], index=1 if init_sex == "Male" else 0, horizontal=True)
            sex_val = 1 if sex_label == "Male" else 0
        with c2:
            gen_hlth_label = st.select_slider("Self-Rated General Health", options=list(GEN_HLTH_MAP.keys()), value=init_hlth)
            gen_hlth_val = GEN_HLTH_MAP[gen_hlth_label]
            diff_walk = st.checkbox("Difficulty walking or climbing stairs", value=init_diff)
            diff_walk_val = 1 if diff_walk else 0
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">2. Clinical & Anthropometric Indicators</div>', unsafe_allow_html=True)
        
        bmi_tab1, bmi_tab2 = st.tabs(["Enter BMI Directly", "Calculate from Height & Weight"])
        with bmi_tab1:
            bmi_input = st.number_input("Body Mass Index (BMI)", min_value=12.0, max_value=65.0, value=float(init_bmi), step=0.1)
        with bmi_tab2:
            calc_c1, calc_c2 = st.columns(2)
            with calc_c1:
                weight_kg = st.number_input("Weight (kg)", min_value=35.0, max_value=220.0, value=85.0, step=0.5)
            with calc_c2:
                height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=175.0, step=1.0)
            computed_bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
            use_computed = st.checkbox(f"Apply calculated BMI ({computed_bmi})", value=False)
            if use_computed:
                bmi_input = computed_bmi
                
        c_vitals1, c_vitals2 = st.columns(2)
        with c_vitals1:
            high_bp = st.checkbox("Diagnosed High Blood Pressure (Hypertension)", value=init_bp)
            high_bp_val = 1 if high_bp else 0
            high_chol = st.checkbox("Diagnosed High Blood Cholesterol", value=init_chol)
            high_chol_val = 1 if high_chol else 0
        with c_vitals2:
            heart_disease = st.checkbox("Coronary Heart Disease or Myocardial Infarction", value=False)
            heart_val = 1 if heart_disease else 0
            stroke = st.checkbox("History of Stroke", value=False)
            stroke_val = 1 if stroke else 0
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">3. Behavioral & Lifestyle Profile</div>', unsafe_allow_html=True)
        c_hab1, c_hab2 = st.columns(2)
        with c_hab1:
            phys_act = st.checkbox("Regular physical activity (past 30 days)", value=init_phys)
            phys_act_val = 1 if phys_act else 0
            smoker = st.checkbox("Lifetime tobacco consumption > 100 cigarettes", value=init_smoke)
            smoker_val = 1 if smoker else 0
        with c_hab2:
            fruit = st.checkbox("Consumes fruit 1+ times daily", value=True)
            fruit_val = 1 if fruit else 0
            veggie = st.checkbox("Consumes vegetables 1+ times daily", value=True)
            veggie_val = 1 if veggie else 0
        st.markdown('</div>', unsafe_allow_html=True)

    # Build evaluation row
    current_patient_dict = {
        "HighBP": high_bp_val,
        "HighChol": high_chol_val,
        "CholCheck": 1,
        "BMI": float(bmi_input),
        "Smoker": smoker_val,
        "Stroke": stroke_val,
        "HeartDiseaseorAttack": heart_val,
        "PhysActivity": phys_act_val,
        "Fruits": fruit_val,
        "Veggies": veggie_val,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": gen_hlth_val,
        "MentHlth": 0,
        "PhysHlth": 0,
        "DiffWalk": diff_walk_val,
        "Sex": sex_val,
        "Age": age_val,
        "Education": 4,
        "Income": 5
    }

    df_eval = pd.DataFrame([current_patient_dict])[feature_names]
    risk_prob = float(model.predict_proba(df_eval)[0, 1])

    with col_results:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Clinical Assessment & Triage Output</div>', unsafe_allow_html=True)
        
        if risk_prob < 0.25:
            tier_class = "tier-low"
            tier_title = "Low Risk Tier"
            tier_msg = "Patient demonstrates favorable metabolic and cardiovascular metrics. Low statistical probability of impaired fasting glucose."
            rec_text = "Standard preventative screening every 3 years. Maintain balanced diet and active routine."
        elif risk_prob < 0.55:
            tier_class = "tier-moderate"
            tier_title = "Moderate / Pre-diabetic Risk Tier"
            tier_msg = "Early metabolic warning signs identified. Elevated risk of progressing to Type 2 Diabetes without intervention."
            rec_text = "Recommend laboratory Fasting Blood Glucose (FBG) or HbA1c screening. Evaluate dietary modifications and structured exercise."
        else:
            tier_class = "tier-high"
            tier_title = "High Clinical Risk Tier"
            tier_msg = "High probability profile. Multiple compounding risk factors present requiring formal clinical diagnostics."
            rec_text = "Schedule comprehensive metabolic panel and diagnostic HbA1c test immediately. Initiate clinical diabetes prevention protocol."

        st.markdown(f"""
        <div class="{tier_class}">
            <span style="font-weight: 600; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.05em;">{tier_title}</span>
            <div class="score-display">{risk_prob * 100:.1f}%</div>
            <p style="margin: 0; font-size: 0.875rem;">{tier_msg}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.progress(min(risk_prob, 1.0))
        
        st.markdown("<h4 style='font-size: 0.95rem; font-weight: 600; color: #1E293B; margin-top: 1.25rem;'>Identified Clinical Drivers</h4>", unsafe_allow_html=True)
        
        flags = []
        if high_bp_val == 1:
            flags.append(("Hypertension Diagnosed", "Hypertension increases vascular resistance and strongly associates with insulin resistance."))
        if high_chol_val == 1:
            flags.append(("Dyslipidemia (High Cholesterol)", "Impaired lipid metabolism is a hallmark of metabolic syndrome."))
        if bmi_input >= 30.0:
            flags.append((f"Obesity Class I/II (BMI {bmi_input:.1f})", "Excess adiposity significantly impairs insulin sensitivity."))
        elif bmi_input >= 25.0:
            flags.append((f"Overweight Threshold (BMI {bmi_input:.1f})", "Moderate contributor to metabolic strain."))
        if phys_act_val == 0:
            flags.append(("Physical Inactivity", "Sedentary lifestyle reduces skeletal muscle glucose uptake."))
        if smoker_val == 1:
            flags.append(("Tobacco Consumption", "Nicotine and oxidative stress impair insulin secretion and vascular health."))
            
        if flags:
            for title, desc in flags:
                st.markdown(f"**• {title}:** <span style='color: #475569; font-size: 0.875rem;'>{desc}</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: #15803D; font-size: 0.875rem;'>No major cardiovascular or metabolic flags identified.</span>", unsafe_allow_html=True)
            
        st.divider()
        st.markdown("<h4 style='font-size: 0.95rem; font-weight: 600; color: #1E293B;'>Clinical Action Protocol</h4>", unsafe_allow_html=True)
        st.info(rec_text)
        st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 2: COUNTERFACTUAL SIMULATOR =================
with tabs[1]:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Actionable Counterfactual Simulation</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #475569; font-size: 0.9rem; margin-bottom: 1.2rem;">
        Assess the prospective clinical impact of non-pharmacological and pharmacological interventions. 
        Modify the parameters below to compute the projected risk reduction relative to the baseline profile.
    </p>
    """, unsafe_allow_html=True)
    
    sim_col1, sim_col2 = st.columns(2, gap="large")
    
    with sim_col1:
        st.markdown("##### Planned Interventions")
        sim_bmi_delta = st.slider("Target BMI Reduction (points)", 0.0, 10.0, 3.5, step=0.5)
        sim_target_bp = st.checkbox("Achieve Blood Pressure Control (< 130/80 mmHg)", value=True if high_bp_val == 1 else False)
        sim_target_chol = st.checkbox("Achieve Lipid Target / Statins Therapy", value=True if high_chol_val == 1 else False)
        sim_target_exercise = st.checkbox("Adopt Guideline Physical Activity (150+ mins/week)", value=True)
        sim_target_smoking = st.checkbox("Smoking Cessation / Abstinence", value=True if smoker_val == 1 else False)
        sim_target_health = st.selectbox("Anticipated General Health State", ["Good", "Very Good", "Excellent"], index=1)
        
    target_dict = current_patient_dict.copy()
    target_dict["BMI"] = max(18.5, float(current_patient_dict["BMI"] - sim_bmi_delta))
    if sim_target_bp:
        target_dict["HighBP"] = 0
    if sim_target_chol:
        target_dict["HighChol"] = 0
    if sim_target_exercise:
        target_dict["PhysActivity"] = 1
    if sim_target_smoking:
        target_dict["Smoker"] = 0
    target_dict["GenHlth"] = GEN_HLTH_MAP[sim_target_health]
    
    df_sim = pd.DataFrame([target_dict])[feature_names]
    sim_prob = float(model.predict_proba(df_sim)[0, 1])
    
    abs_reduction = max(0.0, risk_prob - sim_prob)
    rel_reduction = (abs_reduction / risk_prob * 100) if risk_prob > 0 else 0.0

    with sim_col2:
        st.markdown("##### Projected Risk Trajectory")
        m1, m2 = st.columns(2)
        m1.metric("Baseline Assessment", f"{risk_prob * 100:.1f}%")
        m2.metric("Projected Post-Intervention", f"{sim_prob * 100:.1f}%", delta=f"-{abs_reduction * 100:.1f}%", delta_color="inverse")
        
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 6px; padding: 1.25rem; margin-top: 1rem;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #334155; text-transform: uppercase;">Estimated Relative Benefit</div>
            <div style="font-size: 2rem; font-weight: 700; color: #0F766E; margin: 0.25rem 0;">
                {rel_reduction:.1f}% Risk Reduction
            </div>
            <p style="font-size: 0.875rem; color: #475569; margin: 0; line-height: 1.5;">
                By addressing cardiovascular risk factors and reducing BMI from <strong>{bmi_input:.1f}</strong> to <strong>{target_dict['BMI']:.1f}</strong>,
                the patient shifts into a significantly more resilient metabolic state.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 3: EXPLAINABILITY & DRIVERS =================
with tabs[2]:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Model Interpretability & Epidemiological Evidence</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #475569; font-size: 0.9rem;">
        Feature importance extracted from 253,680 patient encounters within the CDC BRFSS study.
        Trees prioritize clinical indicators that yield maximum information gain in separating normoglycemic vs. dysglycemic profiles.
    </p>
    """, unsafe_allow_html=True)
    
    imp_df = pd.DataFrame(list(feature_importances.items()), columns=["Clinical Indicator", "Information Gain Weight"])
    imp_df = imp_df.sort_values(by="Information Gain Weight", ascending=True).tail(10)
    
    st.bar_chart(data=imp_df.set_index("Clinical Indicator"), horizontal=True, color="#0F766E")
    
    st.markdown("""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.5rem;">
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 6px;">
            <strong style="color: #0F172A; font-size: 0.9rem;">1. Vascular Resistance (HighBP)</strong>
            <p style="color: #475569; font-size: 0.85rem; margin-top: 0.35rem;">
                Hypertension carries over 50% of model split weight, corroborating clinical literature linking arterial stiffness to impaired glucose transport.
            </p>
        </div>
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 6px;">
            <strong style="color: #0F172A; font-size: 0.9rem;">2. Subjective Vitality (GenHlth)</strong>
            <p style="color: #475569; font-size: 0.85rem; margin-top: 0.35rem;">
                Self-rated health serves as an accurate composite marker for systemic inflammation, lethargy, and subclinical metabolic burden.
            </p>
        </div>
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 6px;">
            <strong style="color: #0F172A; font-size: 0.9rem;">3. Lipid Profile (HighChol)</strong>
            <p style="color: #475569; font-size: 0.85rem; margin-top: 0.35rem;">
                Elevated low-density lipoproteins and triglycerides frequently cluster with hyperinsulinemia prior to clinical diabetes diagnosis.
            </p>
        </div>
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 6px;">
            <strong style="color: #0F172A; font-size: 0.9rem;">4. Functional Mobility (DiffWalk)</strong>
            <p style="color: #475569; font-size: 0.85rem; margin-top: 0.35rem;">
                Difficulty with ambulation indicates advanced muscular deconditioning and peripheral microvascular complications.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ================= BATCH TRIAGE HELPERS =================

def build_patient_dict_from_row(row: pd.Series) -> dict:
    """Map CSV column names to model feature names with safe defaults."""
    return {
        "HighBP":              int(row.get("HighBP", 0)),
        "HighChol":            int(row.get("HighChol", 0)),
        "CholCheck":           int(row.get("CholCheck", 1)),
        "BMI":                 float(row.get("BMI", 25.0)),
        "Smoker":              int(row.get("Smoker", 0)),
        "Stroke":              int(row.get("Stroke", 0)),
        "HeartDiseaseorAttack":int(row.get("HeartDiseaseorAttack", 0)),
        "PhysActivity":        int(row.get("PhysActivity", 1)),
        "Fruits":              int(row.get("Fruits", 1)),
        "Veggies":             int(row.get("Veggies", 1)),
        "HvyAlcoholConsump":   int(row.get("HvyAlcoholConsump", 0)),
        "AnyHealthcare":       int(row.get("AnyHealthcare", 1)),
        "NoDocbcCost":         int(row.get("NoDocbcCost", 0)),
        "GenHlth":             int(row.get("GenHlth", 3)),
        "MentHlth":            int(row.get("MentHlth", 0)),
        "PhysHlth":            int(row.get("PhysHlth", 0)),
        "DiffWalk":            int(row.get("DiffWalk", 0)),
        "Sex":                 int(row.get("Sex", 0)),
        "Age":                 int(row.get("Age", 7)),
        "Education":           int(row.get("Education", 4)),
        "Income":              int(row.get("Income", 5)),
    }

def tier_from_prob(prob: float) -> str:
    if prob < 0.25:   return "Low Risk"
    if prob < 0.55:   return "Moderate Risk"
    return "High Risk"

def row_color(tier: str) -> str:
    return {"Low Risk": "#ECFDF5", "Moderate Risk": "#FFFBEB", "High Risk": "#FEF2F2"}.get(tier, "#FFFFFF")

# ================= TAB 4: BATCH COHORT SCREENING =================
with tabs[3]:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Batch Cohort Screening — CSV Upload</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #475569; font-size: 0.9rem; margin-bottom: 1rem;">
        Upload a patient cohort CSV containing any subset of the 21 clinical indicators.
        The engine scores each record, assigns a risk tier, and produces a downloadable
        triage report with flagged high-risk cases highlighted.
    </p>
    """, unsafe_allow_html=True)

    # Sample template download
    sample_data = pd.DataFrame([{
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 32.4, "Smoker": 1,
        "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 1,
        "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
        "GenHlth": 4, "MentHlth": 5, "PhysHlth": 10, "DiffWalk": 1, "Sex": 1,
        "Age": 9, "Education": 4, "Income": 5
    }, {
        "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 21.8, "Smoker": 0,
        "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
        "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
        "GenHlth": 2, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0,
        "Age": 4, "Education": 5, "Income": 7
    }])
    template_csv = sample_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV Template",
        data=template_csv,
        file_name="diaguard_batch_template.csv",
        mime="text/csv"
    )

    st.divider()
    uploaded_file = st.file_uploader("Upload Patient Cohort CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            cohort_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(cohort_df):,} patient records successfully.")

            # Score all rows
            records = []
            for idx, row in cohort_df.iterrows():
                p = build_patient_dict_from_row(row)
                df_row = pd.DataFrame([p])[feature_names]
                prob = float(model.predict_proba(df_row)[0, 1])
                tier = tier_from_prob(prob)
                records.append({
                    "Record #":        idx + 1,
                    "Risk Score (%)":  round(prob * 100, 1),
                    "Risk Tier":       tier,
                    "BMI":             p["BMI"],
                    "High BP":         "Yes" if p["HighBP"] else "No",
                    "High Cholesterol":    "Yes" if p["HighChol"] else "No",
                    "Physically Active":   "Yes" if p["PhysActivity"] else "No",
                    "Smoker":          "Yes" if p["Smoker"] else "No",
                })

            results_df = pd.DataFrame(records).sort_values("Risk Score (%)", ascending=False)

            # Summary metrics
            n_high     = (results_df["Risk Tier"] == "High Risk").sum()
            n_moderate = (results_df["Risk Tier"] == "Moderate Risk").sum()
            n_low      = (results_df["Risk Tier"] == "Low Risk").sum()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Patients",   f"{len(results_df):,}")
            m2.metric("High Risk",        str(n_high),     delta=f"{n_high/len(results_df)*100:.1f}% of cohort", delta_color="inverse")
            m3.metric("Moderate Risk",    str(n_moderate))
            m4.metric("Low Risk",         str(n_low))

            st.markdown("#### Triage Results — Sorted by Risk Score")

            # Colour-coded table via HTML
            rows_html = ""
            for _, r in results_df.iterrows():
                bg = row_color(r["Risk Tier"])
                tier_bold = f"<strong>{r['Risk Tier']}</strong>"
                rows_html += f"""
                <tr style="background-color:{bg};">
                    <td style="padding:0.45rem 0.75rem;">{r['Record #']}</td>
                    <td style="padding:0.45rem 0.75rem; font-weight:600;">{r['Risk Score (%)']:.1f}%</td>
                    <td style="padding:0.45rem 0.75rem;">{tier_bold}</td>
                    <td style="padding:0.45rem 0.75rem;">{r['BMI']}</td>
                    <td style="padding:0.45rem 0.75rem;">{r['High BP']}</td>
                    <td style="padding:0.45rem 0.75rem;">{r['High Cholesterol']}</td>
                    <td style="padding:0.45rem 0.75rem;">{r['Physically Active']}</td>
                    <td style="padding:0.45rem 0.75rem;">{r['Smoker']}</td>
                </tr>"""

            table_html = f"""
            <div style="overflow-x: auto;">
            <table style="width:100%; border-collapse: collapse; font-size: 0.875rem; border: 1px solid #E2E8F0; border-radius: 6px; overflow: hidden;">
                <thead>
                    <tr style="background-color: #F1F5F9; color: #334155;">
                        <th style="padding:0.5rem 0.75rem; text-align:left;">Record</th>
                        <th style="padding:0.5rem 0.75rem; text-align:left;">Risk Score</th>
                        <th style="padding:0.5rem 0.75rem; text-align:left;">Risk Tier</th>
                        <th style="padding:0.5rem 0.75rem; text-align:left;">BMI</th>
                        <th style="padding:0.5rem 0.75rem; text-align:left;">High BP</th>
                        <th style="padding:0.5rem 0.75rem; text-align:left;">High Cholesterol</th>
                        <th style="padding:0.5rem 0.75rem; text-align:left;">Active</th>
                        <th style="padding:0.5rem 0.75rem; text-align:left;">Smoker</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            </div>"""
            st.markdown(table_html, unsafe_allow_html=True)

            # Export triage report
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            export_csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Triage Report (CSV)",
                data=export_csv,
                file_name=f"diaguard_triage_{datetime.date.today().isoformat()}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Failed to process file: {e}")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2.5rem 1rem; background-color: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 8px; color: #94A3B8;">
            <div style="font-size: 0.95rem; font-weight: 500;">No file uploaded yet</div>
            <div style="font-size: 0.85rem; margin-top: 0.35rem;">Download the CSV template above, populate it with patient data, and upload it here.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 5: EHR CONSULTATION NOTE =================
with tabs[4]:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">EHR Consultation Note Generator</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #475569; font-size: 0.9rem; margin-bottom: 1rem;">
        Generates a structured clinical consultation note for the active patient assessment
        (from Tab 1). The output is formatted for direct copy-paste into EHR systems such as Epic,
        Cerner, or any structured clinical documentation platform.
    </p>
    """, unsafe_allow_html=True)

    # Build active patient summary from current session
    case_id    = str(uuid.uuid4())[:8].upper()
    timestamp  = datetime.datetime.now().strftime("%B %d, %Y  %H:%M")
    risk_label = tier_from_prob(risk_prob)

    active_flags_lines = []
    if high_bp_val == 1:
        active_flags_lines.append("  - Hypertension (Diagnosed)")
    if high_chol_val == 1:
        active_flags_lines.append("  - Dyslipidemia / High Cholesterol (Diagnosed)")
    if bmi_input >= 30.0:
        active_flags_lines.append(f"  - Obesity Class I/II  (BMI {bmi_input:.1f})")
    elif bmi_input >= 25.0:
        active_flags_lines.append(f"  - Overweight  (BMI {bmi_input:.1f})")
    if phys_act_val == 0:
        active_flags_lines.append("  - Physical Inactivity (< 150 mins/week)")
    if smoker_val == 1:
        active_flags_lines.append("  - Tobacco Consumption History (> 100 cigarettes lifetime)")
    if diff_walk_val == 1:
        active_flags_lines.append("  - Functional Mobility Limitation (Difficulty Walking)")
    if not active_flags_lines:
        active_flags_lines.append("  - No major cardiovascular or metabolic flags identified")

    if risk_prob < 0.25:
        rec_protocol = (
            "Standard preventative metabolic screening every 3 years. "
            "Reinforce dietary balance and maintenance of current physical activity. "
            "No immediate laboratory workup indicated."
        )
    elif risk_prob < 0.55:
        rec_protocol = (
            "Order Fasting Blood Glucose (FBG) and HbA1c laboratory panel within 30 days. "
            "Refer patient to registered dietitian for Medical Nutrition Therapy (MNT). "
            "Prescribe structured aerobic exercise programme (150 mins/week moderate intensity). "
            "Schedule 6-month follow-up for repeat metabolic assessment."
        )
    else:
        rec_protocol = (
            "Order comprehensive metabolic panel (CMP), Fasting Blood Glucose, and HbA1c immediately. "
            "Consider 2-hour oral glucose tolerance test (OGTT) if initial results are borderline. "
            "Initiate CDC-recognised Diabetes Prevention Programme (DPP) referral. "
            "Evaluate antihypertensive therapy adjustment if BP remains uncontrolled. "
            "Schedule endocrinology consultation and 3-month follow-up."
        )

    flags_block = "\n".join(active_flags_lines)

    note = f"""================================================================================
  CLINICAL RISK CONSULTATION NOTE  —  DiaGuard CDS Platform v1.0
================================================================================

  Case Reference ID  : DG-{case_id}
  Assessment Date    : {timestamp}
  Attending Provider : [Clinician Name / Registration No.]
  Facility           : [Hospital / Clinic Name]

────────────────────────────────────────────────────────────────────────────────
  SECTION 1 — PATIENT DEMOGRAPHICS
────────────────────────────────────────────────────────────────────────────────

  Age Category       : {age_label}
  Biological Sex     : {sex_label}
  BMI                : {bmi_input:.1f}  kg/m²
  General Health     : {gen_hlth_label}

────────────────────────────────────────────────────────────────────────────────
  SECTION 2 — AI-ASSISTED RISK STRATIFICATION
────────────────────────────────────────────────────────────────────────────────

  Algorithm          : XGBoost Gradient Boosting Classifier
  Training Cohort    : CDC BRFSS (n = 253,680 patient encounters)
  Validation AUC     : 0.8272  |  Clinical Sensitivity: 79.59%

  Predicted Risk Probability   : {risk_prob * 100:.1f}%
  Clinical Risk Tier           : {risk_label.upper()}

────────────────────────────────────────────────────────────────────────────────
  SECTION 3 — IDENTIFIED CLINICAL RISK DRIVERS
────────────────────────────────────────────────────────────────────────────────

{flags_block}

────────────────────────────────────────────────────────────────────────────────
  SECTION 4 — RECOMMENDED CLINICAL ACTION PROTOCOL
────────────────────────────────────────────────────────────────────────────────

  {rec_protocol}

────────────────────────────────────────────────────────────────────────────────
  SECTION 5 — CLINICIAN ATTESTATION
────────────────────────────────────────────────────────────────────────────────

  This AI-generated risk stratification is intended as clinical decision
  support only. It does not constitute a definitive medical diagnosis.
  Final clinical judgement rests with the attending licensed practitioner.

  Clinician Signature : ______________________________
  Date                : ______________________________
  Licence / Reg. No.  : ______________________________

================================================================================
"""

    st.markdown("#### Live Preview")
    st.code(note, language=None)

    st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="Download Note (.txt)",
            data=note.encode("utf-8"),
            file_name=f"DiaGuard_EHR_Note_DG-{case_id}.txt",
            mime="text/plain"
        )
    with dl_col2:
        st.download_button(
            label="Download Note (.md)",
            data=note.encode("utf-8"),
            file_name=f"DiaGuard_EHR_Note_DG-{case_id}.md",
            mime="text/markdown"
        )

    st.caption(
        "Note: Re-run the patient assessment in Tab 1 to refresh this note "
        "for a different patient profile."
    )
    st.markdown('</div>', unsafe_allow_html=True)
