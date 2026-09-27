# 🛡️ DiaGuard: Diabetes Risk Assessment & Counterfactual Guidance

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB5424?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)

**DiaGuard** is an end-to-end, production-grade Machine Learning system designed for **population-level diabetes risk assessment and personalized lifestyle guidance**. 

Trained on the official **CDC Behavioral Risk Factor Surveillance System (BRFSS)** dataset of **253,680 patient records**, DiaGuard couples high-recall predictive modeling with **counterfactual simulation**—empowering patients and clinicians to see how specific lifestyle modifications (such as reducing BMI, controlling blood pressure, or adopting routine exercise) directly lower diabetes risk.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Data_Pipeline["1. Ingestion & Preprocessing"]
        CDC["CDC BRFSS Dataset<br/>(253,680 records, 21 indicators)"] --> Ingestion["Automated Ingestion<br/>src/data_loader.py"]
        Ingestion --> Split["Stratified 80/20 Train-Test Split"]
    end

    subgraph Model_Engine["2. Machine Learning Engine"]
        Split --> XGB["XGBoost Classifier<br/>(scale_pos_weight = 6.18)"]
        XGB --> Eval["Model Evaluation<br/>ROC-AUC: 0.8272 | Recall: 79.6%"]
        Eval --> Bundle["Model Package<br/>models/diaguard_model.joblib"]
    end

    subgraph Backend_Serving["3. Serving & Microservices"]
        Bundle --> API["FastAPI Inference Engine<br/>src/api.py"]
        API --> Docs["Interactive Swagger Docs<br/>http://localhost:8000/docs"]
    end

    subgraph User_Experience["4. Clinical Web Interface"]
        Bundle --> UI["Streamlit Dashboard<br/>src/app.py"]
        UI --> RiskGauge["Risk Triage Meter (Low/Mod/High)"]
        UI --> WhatIf["'What-If' Lifestyle Simulator"]
        UI --> XAI["CDC Feature Importance Charts"]
    end

    subgraph Observability_CICD["5. MLOps Observability & CI/CD"]
        API --> Drift["Kolmogorov-Smirnov Drift Monitor<br/>src/drift_monitor.py"]
        CI["GitHub Actions CI Workflow<br/>.github/workflows/ci.yml"] --> Tests["Pytest Pipeline<br/>tests/test_pipeline.py"]
    end
```

---

## 📊 Dataset & Model Performance

### Dataset Overview
* **Source:** Centers for Disease Control and Prevention (CDC) — Behavioral Risk Factor Surveillance System (BRFSS)
* **Sample Size:** 253,680 patient survey encounters
* **Features:** 21 physiological, demographic, and behavioral health indicators
* **Class Imbalance:** ~14% positive cases (handled via cost-sensitive learning with `scale_pos_weight = 6.18`)

### Test Set Performance (50,736 Holdout Patients)

| Metric | Score | Clinical Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **`0.8272`** | Strong discrimination between healthy individuals and diabetic/pre-diabetic cases |
| **Recall (Sensitivity)** | **`79.59%`** | **Critical for healthcare screening:** Catches ~8 out of 10 at-risk individuals |
| **Accuracy** | **`72.14%`** | Calibrated screening baseline on real-world imbalanced population |
| **Latency** | **`< 25 ms`** | Real-time scoring suitable for instant web and mobile triage |

---

## 🌟 Key Features

1. **Patient Risk Scoring (`/api/v1/predict`):**
   * Inputs 21 indicators and computes calibrated risk probability.
   * Classifies individuals into 4 clinical tiers: *Low Risk*, *Moderate Risk*, *Elevated Risk*, and *High Risk*.
2. **Counterfactual "What-If" Guidance (`/api/v1/simulate-what-if`):**
   * Computes the exact relative and absolute risk reduction achievable when a patient modifies their BMI, adopts exercise, or controls blood pressure.
3. **Statistical Drift Observability (`src/drift_monitor.py`):**
   * Automatically executes 2-sample Kolmogorov-Smirnov tests to detect demographic and physiological data drift in incoming patient batches.
4. **Production Architecture:**
   * Pydantic data schemas rejecting invalid vitals (e.g. negative BMI or out-of-range ages).
   * Fully containerized with Docker & Docker Compose.
   * Automated unit and integration testing with `pytest`.

---

## 🚀 Quickstart & How to Run

### Option 1: Native Local Run (Recommended for Development)

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Data & Train Model (Runs in ~20 seconds):**
   ```bash
   python src/data_loader.py
   python src/train.py
   ```

3. **Run the Interactive Web App:**
   ```bash
   streamlit run src/app.py
   ```
   *Open your browser at `http://localhost:8501`*

4. **Run the FastAPI Backend Server:**
   ```bash
   uvicorn src.api:app --reload --port 8000
   ```
   *Explore the interactive API docs at `http://localhost:8000/docs`*

5. **Run Automated Tests & Drift Monitor:**
   ```bash
   pytest tests/ -v
   python src/drift_monitor.py
   ```

---

### Option 2: Docker Compose Deployment

Run the entire multi-service stack with a single command:
```bash
docker compose up --build
```
* **Frontend Web App:** `http://localhost:8501`
* **FastAPI Docs:** `http://localhost:8000/docs`

---

## 📁 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI/CD test pipeline
├── data/
│   └── raw/                     # Versioned CDC epidemiological dataset
├── models/
│   ├── diaguard_model.joblib    # Serialized XGBoost model bundle (327 KB)
│   ├── evaluation_metrics.json  # Holdout test metrics
│   └── drift_report.json        # Data drift monitoring logs
├── src/
│   ├── data_loader.py           # Automated dataset fetcher (UCI repo)
│   ├── train.py                 # Training script with class-weighting & evaluation
│   ├── api.py                   # FastAPI REST backend with Pydantic validation
│   ├── app.py                   # Streamlit web application & What-If simulator
│   └── drift_monitor.py         # Kolmogorov-Smirnov drift detection service
├── tests/
│   └── test_pipeline.py         # Pytest unit & integration test suite
├── Dockerfile                   # Multi-service container specification
├── docker-compose.yml           # Local multi-container orchestration
├── requirements.txt             # Pinned project dependencies
└── README.md                    # Project documentation
```

---

## 👩‍💻 Author
**Anya Kumar**  
B.Tech in Computer Science & Engineering (AI/ML Specialization)
