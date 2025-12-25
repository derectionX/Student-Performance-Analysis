# Student Performance Predictor

AI-powered Streamlit app for analyzing and predicting student exam scores using a trained Random Forest model.

## Features
- Individual student prediction form
- Batch CSV upload and prediction
- Visual analytics (distribution, metrics)
- Model insights (feature importance)

## Environment
- OS: Windows (PowerShell)
- Python: 3.8+

## Setup
1) Create and activate a virtual environment

```powershell path=null start=null
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell path=null start=null
pip install -r requirements.txt
```

## Run the app

```powershell path=null start=null
streamlit run app.py
```

## Model artifact
- The app expects a trained model at `student_model.pkl`.
- The model’s expected input columns (after one-hot encoding with `drop_first=True`) include:
  - Numeric: Hours_Studied, Attendance, Sleep_Hours, Previous_Scores, Tutoring_Sessions, Physical_Activity
  - Categorical (examples of dummified columns):
    - Parental_Involvement_Low, Parental_Involvement_Medium (High is baseline)
    - Access_to_Resources_Low, Access_to_Resources_Medium (High is baseline)
    - Extracurricular_Activities_Yes (No is baseline)
    - Motivation_Level_Low, Motivation_Level_Medium (High is baseline)
    - Internet_Access_Yes (No is baseline)
    - Family_Income_Low, Family_Income_Medium (High is baseline)
    - Teacher_Quality_Low, Teacher_Quality_Medium (High is baseline)
    - School_Type_Public (Private is baseline)
    - Peer_Influence_Neutral, Peer_Influence_Positive (Negative is baseline)
    - Learning_Disabilities_Yes (No is baseline)
    - Parental_Education_Level_High School, Parental_Education_Level_Postgraduate (College is baseline)
    - Distance_from_Home_Near, Distance_from_Home_Moderate (Far is baseline)
    - Gender_Male (Female is baseline)

## Batch CSV schema
Prepare your CSV with exactly these columns (headers case-sensitive):

- Hours_Studied (number)
- Attendance (number)
- Parental_Involvement (Low | Medium | High)
- Access_to_Resources (Low | Medium | High)
- Extracurricular_Activities (Yes | No)
- Sleep_Hours (number)
- Previous_Scores (number)
- Motivation_Level (Low | Medium | High)
- Internet_Access (Yes | No)
- Tutoring_Sessions (number)
- Family_Income (Low | Medium | High)
- Teacher_Quality (Low | Medium | High)
- School_Type (Public | Private)
- Peer_Influence (Negative | Neutral | Positive)
- Physical_Activity (number)
- Learning_Disabilities (Yes | No)
- Parental_Education_Level (High School | College | Postgraduate)
- Distance_from_Home (Near | Moderate | Far)
- Gender (Male | Female)

Tip: Download the provided `sample_students_template.csv` from within the app’s Batch Analysis tab.

## Notebooks
- `app.ipynb`, `main1.ipynb` likely contain data exploration or training steps. If you want a CLI training script, we can extract the logic into a `train.py` on request.

## Notes
- The app includes validation for batch CSV column names.
- If the model file is missing or corrupted, the app will show an error and stop gracefully.
