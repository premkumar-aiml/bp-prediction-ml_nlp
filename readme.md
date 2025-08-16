## Overview

This project aims to build an end-to-end system that predicts **Diastolic Blood Pressure (DBP)** using machine learning techniques and provides **personalized intervention suggestions** by processing biomedical literature (PubMed articles) using NLP methods.

The approach integrates:
- **ML models** for DBP prediction based on patient data (age, gender, lifestyle, etc.)
- **NLP techniques** to extract patient-specific advice from biomedical texts for early risk mitigation and health guidance.


## Objectives

- Predict diastolic blood pressure using patient demographic and clinical data.
- Analyze and evaluate different regression models (e.g., Linear Regression, Random Forest, KNN).
- Process locally stored PubMed literature using NLP for personalized intervention retrieval.
- Match predicted BP values and patient profiles to relevant medical recommendations using similarity-based techniques.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/premkumar-aiml/bp-prediction-ml_nlp.git

Create a virtual environment and activate it:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows