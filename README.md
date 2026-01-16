# 🧠 DeciSense AI  
### Growth & Workforce Decision Intelligence Platform (MVP)

🔗 **Live Demo**  
👉 https://decisense-ai-f5sy3wegcwdvvmsxtcupv4.streamlit.app/

---

## 📌 Overview

**DeciSense AI** is a publicly deployed **Decision Intelligence** platform that helps organisations and founders make better strategic decisions by combining:

- predictive machine learning,
- scenario simulation,
- explainable recommendations, and
- trade-off–aware decision ranking.

Unlike traditional analytics projects that stop at prediction, DeciSense AI focuses on **what decision should be taken next**, making it directly usable in real business contexts.

This project was independently designed, built, and deployed as a **Minimum Viable Product (MVP)**.

---

## 🎯 Problem Statement

Organisations often struggle with questions such as:

- Should we invest more in growth or retention?
- How will workforce decisions affect business outcomes?
- What trade-offs exist between cost, growth, and attrition?
- Which strategic option is actually best?

Most machine learning projects answer *what might happen*.  
**DeciSense AI answers what should be done.**

---

## 💡 Solution: Decision Intelligence

DeciSense AI adopts a **decision-centric approach**:

1. Predicts outcomes (Growth & Attrition)
2. Simulates multiple strategic scenarios
3. Quantifies trade-offs and uncertainty
4. Ranks decisions using a utility-based framework
5. Explains *why* a recommendation is made

This positions the project within the growing field of **Decision Intelligence**, beyond standard predictive analytics.

---

## ⚙️ Key Features

- 📈 Growth Probability Modelling  
- 👥 Attrition Risk Prediction  
- 🔁 Scenario Simulation (marketing, hiring, retention strategies)  
- 🧮 Utility-Based Scenario Ranking  
- 🧠 Explainable Recommendations  
- 📊 Visual Trade-off Analysis  
- 📥 Exportable Outputs  
  - CSV (data)
  - PDF decision report (interpretation + rationale)

---

## 🧪 Example Use Cases

- Workforce planning and retention strategy
- Business growth prioritisation
- Resource allocation decisions
- Early-stage startup decision support
- Strategic scenario comparison for leadership teams

---

## 🏗️ Architecture Overview

User Inputs (Business & Workforce Signals)
↓
Scenario Generator
↓
ML Pipelines (Growth & Attrition Models)
↓
Scenario Evaluation & Ranking
↓
Explainable Output + Visualisation
↓
CSV / PDF Decision Reports
---

## 🧠 Models Used

- Supervised classification models using scikit-learn pipelines
- Separate pipelines for:
  - Growth likelihood
  - Attrition risk
- Feature engineering via categorical encoding and scaling
- Models stored as reusable `.joblib` pipelines

> Note: Synthetic data is used to enable public deployment while preserving realistic business patterns.

---

## 🖥️ Live Application

The application is deployed using **Streamlit Cloud** and is fully interactive.

🔗 **Try the app here:**  
https://decisense-ai-f5sy3wegcwdvvmsxtcupv4.streamlit.app/

Users can:
- enter business context and constraints,
- generate ranked strategic recommendations,
- understand what each metric means,
- export results for further use.

---

## 📁 Repository Structure
decisense-ai/
├── app/
│ └── streamlit_app.py
├── src/
│ └── decision/
│ ├── scenario_simulator.py
│ └── ranker.py
├── models/
│ ├── growth_model.joblib
│ └── attrition_model.joblib
├── data/
│ └── synthetic_business_workforce.csv
├── assets/
│ └── metrics & plots
├── requirements.txt
└── README.md

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py

## 📊 Outputs

Ranked decision scenarios

Growth vs attrition trade-off visualisation

CSV export for analysis

PDF decision report including:

input summary

explanation of metrics

top recommendation rationale
