# Apex Financial - Loan Default Prediction

A modern fintech web application for predicting loan default risk using Machine Learning.

## Project Structure
- `frontend/`: Streamlit web interface
- `backend/`: Flask REST API and ML model

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate the dummy Machine Learning model (required before starting backend):
```bash
python backend/train_dummy_model.py
```

3. Start the Flask Backend (Terminal 1):
```bash
python backend/app.py
```
The API will run at http://127.0.0.1:5000

4. Start the Streamlit Frontend (Terminal 2):
```bash
streamlit run frontend/app.py
```
The UI will run at http://localhost:8501
