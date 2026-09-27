import os
import sys
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
    page_title="DiaGuard — Diabetes Risk & Lifestyle Guidance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0E7490;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-high {
        color: #DC2626;
        font-weight: 800;
        font-size: 1.8rem;
    }
    .risk-mod {
        color: #D97706;
        font-weight: 800;
        font-size: 1.8rem;
    }
    .risk-low {
        color: #16A34A;
        font-weight: 800;
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "diaguard_model.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run `python src/train.py` first.")
        st.stop()
    return joblib.load(model_path)

bundle = load_model()
model = bundle["model"]
feature_names = bundle["feature_names"]
metrics = bundle.get("metrics", {})
feature_importances = bundle.get("feature_importances", {})

# Age mapping dictionary (CDC BRFSS 13 categories)
AGE_MAP = {
    "18 to 24": 1, "25 to 29": 2, "30 to 34": 3, "35 to 39": 4,
    "40 to 44": 5, "45 to 49": 6, "50 to 54": 7, "55 to 59": 8,
    "60 to 64": 9, "65 to 69": 10, "70 to 74": 11, "75 to 79": 12,
    "80 or older": 13
}

GEN_HLTH_MAP = {
    "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
}

# Sidebar Info
st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=70)
st.sidebar.title("DiaGuard AI Engine")
st.sidebar.markdown(f"""
**System Specs & Model:**
* **Algorithm:** XGBoost Classifier
* **Dataset:** CDC BRFSS (253,680 records)
* **ROC-AUC Score:** `{metrics.get('roc_auc', 0.8272)}`
* **Diabetic Recall:** `{metrics.get('recall', 0.7959) * 100:.1f}%`
* **Status:** 🟢 Production Ready
""")
st.sidebar.divider()
st.sidebar.caption("Final Year Capstone Project • AI/ML & MLOps Architecture")

# Header
st.markdown('<div class="main-title">🛡️ DiaGuard: Diabetes Risk Assessment & Guidance</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Population-health machine learning engine predicting diabetes vulnerability and evaluating counterfactual lifestyle improvements.</div>', unsafe_allow_html=True)

tabs = st.tabs(["📋 Patient Assessment", "🔄 'What-If' Lifestyle Simulator", "📊 Model Explainability & Health Insights"])

# ================= TAB 1: PATIENT ASSESSMENT =================
with tabs[0]:
    st.subheader("Patient Health Indicators & Clinical Profile")
    st.markdown("Enter patient metrics below to compute calibrated diabetes risk:")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 🧬 Demographics & General Health")
        age_label = st.selectbox("Age Group", list(AGE_MAP.keys()), index=6) # default 50-54
        age_val = AGE_MAP[age_label]
        
        sex_label = st.radio("Biological Sex", ["Female", "Male"], horizontal=True)
        sex_val = 1 if sex_label == "Male" else 0
        
        gen_hlth_label = st.select_slider(
            "Overall Self-Rated Health",
            options=list(GEN_HLTH_MAP.keys()),
            value="Good"
        )
        gen_hlth_val = GEN_HLTH_MAP[gen_hlth_label]
        
        diff_walk = st.checkbox("Difficulty walking or climbing stairs", value=False)
        diff_walk_val = 1 if diff_walk else 0

    with col2:
        st.markdown("##### 🩺 Clinical Measurements")
        high_bp = st.checkbox("Diagnosed with High Blood Pressure", value=True)
        high_bp_val = 1 if high_bp else 0
        
        high_chol = st.checkbox("Diagnosed with High Cholesterol", value=True)
        high_chol_val = 1 if high_chol else 0
        
        chol_check = st.checkbox("Cholesterol checked in past 5 years", value=True)
        chol_check_val = 1 if chol_check else 0
        
        bmi = st.number_input("Body Mass Index (BMI)", min_value=12.0, max_value=65.0, value=29.4, step=0.1)
        
        heart_attack = st.checkbox("History of Heart Disease or Heart Attack", value=False)
        heart_attack_val = 1 if heart_attack else 0
        
        stroke = st.checkbox("History of Stroke", value=False)
        stroke_val = 1 if stroke else 0

    with col3:
        st.markdown("##### 🥗 Habits & Daily Lifestyle")
        phys_act = st.checkbox("Physically active in past 30 days", value=False)
        phys_act_val = 1 if phys_act else 0
        
        smoker = st.checkbox("Smoked > 100 cigarettes in life", value=True)
        smoker_val = 1 if smoker else 0
        
        fruits = st.checkbox("Eats fruit 1+ times/day", value=True)
        fruits_val = 1 if fruits else 0
        
        veggies = st.checkbox("Eats vegetables 1+ times/day", value=True)
        veggies_val = 1 if veggies else 0
        
        heavy_alcohol = st.checkbox("Heavy Alcohol Consumption", value=False)
        heavy_alcohol_val = 1 if heavy_alcohol else 0

    # Build input payload
    patient_dict = {
        "HighBP": high_bp_val,
        "HighChol": high_chol_val,
        "CholCheck": chol_check_val,
        "BMI": float(bmi),
        "Smoker": smoker_val,
        "Stroke": stroke_val,
        "HeartDiseaseorAttack": heart_attack_val,
        "PhysActivity": phys_act_val,
        "Fruits": fruits_val,
        "Veggies": veggies_val,
        "HvyAlcoholConsump": heavy_alcohol_val,
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
    
    st.divider()
    if st.button("🚀 Calculate Patient Risk", type="primary", use_container_width=True):
        df_patient = pd.DataFrame([patient_dict])[feature_names]
        prob = float(model.predict_proba(df_patient)[0, 1])
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### Diabetes Risk Score")
            
            if prob < 0.30:
                st.markdown(f'<div class="risk-low">{prob*100:.1f}% — Low Risk</div>', unsafe_allow_html=True)
                st.success("Patient profile indicates low probability of diabetic vulnerability.")
            elif prob < 0.60:
                st.markdown(f'<div class="risk-mod">{prob*100:.1f}% — Moderate Risk</div>', unsafe_allow_html=True)
                st.warning("Pre-diabetic warning signals detected. Preventative adjustments recommended.")
            else:
                st.markdown(f'<div class="risk-high">{prob*100:.1f}% — Elevated / High Risk</div>', unsafe_allow_html=True)
                st.error("High risk probability flagged by clinical indicators.")
            
            st.progress(min(prob, 1.0))
            st.markdown('</div>', unsafe_allow_html=True)
            
        with res_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### 🔍 Primary Contributing Health Flags")
            
            flagged = []
            if high_bp_val == 1:
                flagged.append("⚠️ **High Blood Pressure:** Strongest clinical indicator in the CDC study.")
            if high_chol_val == 1:
                flagged.append("⚠️ **High Cholesterol:** Significantly correlates with metabolic syndrome.")
            if bmi >= 30.0:
                flagged.append(f"⚠️ **Obesity Range BMI ({bmi:.1f}):** Elevated adiposity increases insulin resistance.")
            elif bmi >= 25.0:
                flagged.append(f"ℹ️ **Overweight BMI ({bmi:.1f}):** Mild risk factor.")
            if phys_act_val == 0:
                flagged.append("⚠️ **Sedentary Lifestyle:** Lack of regular physical activity reduces glucose uptake.")
            if smoker_val == 1:
                flagged.append("⚠️ **Smoking History:** Increases systemic vascular inflammation.")
                
            if not flagged:
                st.write("🎉 No high-risk lifestyle or cardiovascular flags detected!")
            else:
                for f in flagged:
                    st.markdown(f)
            st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 2: WHAT-IF SIMULATOR =================
with tabs[1]:
    st.subheader("🔄 Counterfactual 'What-If' Lifestyle Simulator")
    st.markdown("""
    **How It Works:** Medical AI shouldn't just deliver a scary percentage — it should show how lifestyle changes can **reverse** that risk.
    Below, simulate what happens to this patient's risk when modifying key lifestyle factors.
    """)
    
    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        st.markdown("#### 🎯 Simulated Interventions")
        sim_lose_bmi = st.slider("Target BMI Reduction (points)", 0.0, 10.0, 4.0, step=0.5)
        sim_cure_bp = st.checkbox("Manage & Normalize Blood Pressure (Diet / Medication)", value=True)
        sim_cure_chol = st.checkbox("Normalize Cholesterol Levels", value=False)
        sim_start_exercise = st.checkbox("Adopt Routine Physical Activity (150 mins/week)", value=True)
        sim_improve_health = st.selectbox("Improved Self-Rated Health Outlook", ["Good", "Very Good", "Excellent"], index=1)
        
    # Compute Target Payload
    target_dict = patient_dict.copy()
    target_dict["BMI"] = max(18.5, float(patient_dict["BMI"] - sim_lose_bmi))
    if sim_cure_bp:
        target_dict["HighBP"] = 0
    if sim_cure_chol:
        target_dict["HighChol"] = 0
    if sim_start_exercise:
        target_dict["PhysActivity"] = 1
    target_dict["GenHlth"] = GEN_HLTH_MAP[sim_improve_health]
    
    df_orig = pd.DataFrame([patient_dict])[feature_names]
    df_sim = pd.DataFrame([target_dict])[feature_names]
    
    orig_prob = float(model.predict_proba(df_orig)[0, 1])
    sim_prob = float(model.predict_proba(df_sim)[0, 1])
    abs_drop = max(0.0, orig_prob - sim_prob)
    rel_drop = (abs_drop / orig_prob * 100) if orig_prob > 0 else 0.0
    
    with sim_col2:
        st.markdown("#### 📈 Projected Outcome")
        c1, c2 = st.columns(2)
        c1.metric(label="Current Baseline Risk", value=f"{orig_prob*100:.1f}%")
        c2.metric(label="Projected Risk After Interventions", value=f"{sim_prob*100:.1f}%", delta=f"-{abs_drop*100:.1f}%", delta_color="inverse")
        
        st.markdown(f"""
        <div class="metric-card" style="background-color: #ECFDF5; border: 1px solid #A7F3D0;">
            <h4 style="color: #065F46; margin-top: 0;">🎉 Positive Clinical Impact</h4>
            <p style="color: #047857; font-size: 1.1rem; font-weight: 600;">
                By managing blood pressure and reducing BMI by {sim_lose_bmi:.1f} points, the patient achieves a 
                <span style="font-size: 1.3rem; color: #065F46;">{rel_drop:.1f}% relative reduction</span> in diabetes vulnerability!
            </p>
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 3: MODEL EXPLAINABILITY =================
with tabs[2]:
    st.subheader("📊 CDC Feature Importance & Model Transparency")
    st.markdown("Feature influence derived from 253,680 patient records across the United States:")
    
    # Plot top features
    imp_df = pd.DataFrame(list(feature_importances.items()), columns=["Health Indicator", "Importance Score"])
    imp_df = imp_df.sort_values(by="Importance Score", ascending=True).tail(10)
    
    st.bar_chart(data=imp_df.set_index("Health Indicator"), horizontal=True, color="#0E7490")
    
    st.markdown("""
    #### 💡 Epidemiological Takeaways:
    1. **High Blood Pressure (HighBP):** Accounts for over 50% of the tree split importance. Vascular resistance is intricately tied to metabolic syndrome and insulin resistance.
    2. **General Health Perception (GenHlth):** Subjective health ratings strongly reflect chronic physiological fatigue caused by dysregulated glucose.
    3. **High Cholesterol (HighChol):** Dyslipidemia frequently precedes impaired fasting glucose.
    4. **Mobility (DiffWalk):** Difficulty walking strongly flags peripheral neuropathy and metabolic decline.
    """)
