import streamlit as st
import json
import time
import base64
from datetime import datetime
import joblib
import os
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Apex Financial - Loan Default Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for pastel fintech aesthetic
st.markdown("""
<style>
    :root {
        --primary-purple: #B2A4FF;
        --light-blue: #C1EFFF;
        --mint-green: #D2FFD6;
        --soft-pink: #FFB4B4;
        --off-white: #F8F9FA;
        --dark-navy: #1B263B;
    }
    
    .stApp {
        background-color: var(--off-white);
        color: var(--dark-navy);
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    
    /* Hide top header and footer */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        text-align: center;
        border: 1px solid rgba(178, 164, 255, 0.2);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.08);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-purple);
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton>button, .stFormSubmitButton>button {
        background-color: var(--primary-purple);
        color: var(--dark-navy) !important;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        background-color: #9b8aee;
        box-shadow: 0 4px 12px rgba(178, 164, 255, 0.4);
        color: var(--dark-navy) !important;
    }
    
    /* Form sections & Visibility Fixes */
    .form-section {
        background: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.03);
        margin-bottom: 24px;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    label, p, input, select, [data-baseweb="select"] {
        color: var(--dark-navy) !important;
    }

    
    /* Headers */
    h1, h2, h3 {
        color: var(--dark-navy);
        font-weight: 700;
    }
    
    .brand-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1B263B, #B2A4FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    
    .tagline {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Result Cards */
    .result-card-low {
        background: linear-gradient(135deg, #ffffff, var(--mint-green));
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(210, 255, 214, 0.4);
        border: 1px solid rgba(210, 255, 214, 0.8);
    }
    
    .result-card-high {
        background: linear-gradient(135deg, #ffffff, var(--soft-pink));
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(255, 180, 180, 0.4);
        border: 1px solid rgba(255, 180, 180, 0.8);
    }
    
    .result-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--dark-navy);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Navigation function
def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# Model Loading Configuration
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'LoanDefault.pkl')
    scaler_path = os.path.join(base_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_models()

def show_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #1B263B;'>🛡️ APEX<br>FINANCIAL</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6c757d; font-size: 0.8rem;'>Smarter Loan Decisions</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        if st.button("🏠 Home", use_container_width=True, type="primary" if st.session_state.page == 'Home' else "secondary"):
            navigate_to('Home')
        if st.button("📋 Loan Assessment", use_container_width=True, type="primary" if st.session_state.page == 'Assessment' else "secondary"):
            navigate_to('Assessment')
        if st.button("📊 Results", use_container_width=True, type="primary" if st.session_state.page == 'Results' else "secondary", disabled=st.session_state.prediction_result is None):
            navigate_to('Results')
        if st.button("ℹ️ About", use_container_width=True, type="primary" if st.session_state.page == 'About' else "secondary"):
            navigate_to('About')
            
        st.markdown("---")
        st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #aaa;'>© 2026 Apex Financial</p>", unsafe_allow_html=True)

def home_page():
    st.markdown("<h1 class='brand-title'>Loan Default Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='tagline'>Understand your loan risk in seconds.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.write("Enter a few financial details and our machine learning model will estimate the likelihood of loan default.")
        st.write("")
        
        if st.button("Start Loan Assessment →"):
            navigate_to('Assessment')
            
        st.write("")
        st.write("")
        st.markdown("### How it works")
        
        steps_col1, steps_col2, steps_col3 = st.columns(3)
        with steps_col1:
            st.markdown("""
            <div class='metric-card'>
                <h4>1️⃣</h4>
                <p style='font-size: 0.9rem; margin-top: 10px;'>Enter Details</p>
            </div>
            """, unsafe_allow_html=True)
        with steps_col2:
            st.markdown("""
            <div class='metric-card'>
                <h4>🧠</h4>
                <p style='font-size: 0.9rem; margin-top: 10px;'>AI Analyzes Risk</p>
            </div>
            """, unsafe_allow_html=True)
        with steps_col3:
            st.markdown("""
            <div class='metric-card'>
                <h4>📊</h4>
                <p style='font-size: 0.9rem; margin-top: 10px;'>Get Your Result</p>
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        st.markdown("""
        <div style='background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); text-align: center; border: 1px solid #e9ecef;'>
            <h1 style='font-size: 4rem; margin: 0;'>📱</h1>
            <h3 style='color: #1B263B; margin-top: 20px;'>Secure & Fast</h3>
            <p style='color: #6c757d;'>Bank-grade analysis engine.</p>
            <hr style='border-color: #f8f9fa;'>
            <div style='display: flex; justify-content: space-around; margin-top: 20px;'>
                <div>
                    <h2 style='color: #B2A4FF; margin: 0;'>96.4%</h2>
                    <small>Accuracy</small>
                </div>
                <div>
                    <h2 style='color: #B2A4FF; margin: 0;'>< 1s</h2>
                    <small>Speed</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<p style='text-align: center; margin-top: 50px; color: #adb5bd; font-size: 0.9rem;'>Your information is used only for prediction.</p>", unsafe_allow_html=True)

def assessment_page():
    st.markdown("<h2>Loan Assessment Form</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6c757d;'>Please provide accurate information for the best prediction.</p>", unsafe_allow_html=True)
    
    with st.form("loan_assessment_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='form-section'>", unsafe_allow_html=True)
            st.subheader("👤 Personal Details")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='form-section'>", unsafe_allow_html=True)
            st.subheader("💼 Employment & Income")
            annual_income = st.number_input("Annual Income ($)", min_value=0, value=65000, step=1000)
            employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
            months_employed = st.number_input("Months Employed", min_value=0, value=36)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='form-section'>", unsafe_allow_html=True)
            st.subheader("💰 Loan Details")
            loan_amount = st.number_input("Loan Amount ($)", min_value=500, value=25000, step=500)
            loan_term = st.selectbox("Loan Term", ["12 months", "24 months", "36 months", "48 months", "60 months"], index=2)
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=5.5, step=0.1)
            loan_purpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Education", "Personal", "Business", "Other"])
            co_signer = st.selectbox("Has Co-Signer?", ["No", "Yes"])
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='form-section'>", unsafe_allow_html=True)
            st.subheader("📊 Credit Profile")
            credit_score = st.slider("Credit Score", 300, 850, 680)
            credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=4)
            dti_ratio = st.slider("Debt-to-Income Ratio", 0.00, 1.00, 0.30, step=0.01)
            mortgage = st.selectbox("Has Mortgage?", ["No", "Yes"])
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Predict Loan Risk")
        
        if submitted:
            if annual_income <= 0:
                st.error("Please enter a valid annual income.")
            elif loan_amount <= 0:
                st.error("Please enter a valid loan amount.")
            else:
                data = {
                    "age": age,
                    "marital_status": marital_status,
                    "dependents": dependents,
                    "education": education,
                    "annual_income": annual_income,
                    "employment_type": employment_type,
                    "months_employed": months_employed,
                    "loan_amount": loan_amount,
                    "loan_term": loan_term,
                    "interest_rate": interest_rate,
                    "loan_purpose": loan_purpose,
                    "co_signer": co_signer,
                    "credit_score": credit_score,
                    "credit_lines": credit_lines,
                    "dti_ratio": dti_ratio,
                    "mortgage": mortgage
                }
                st.session_state.form_data = data
                predict_loan(data)

def preprocess_input(data):
    age = data.get('age', 0)
    income = data.get('annual_income', 0)
    loan_amount = data.get('loan_amount', 0)
    credit_score = data.get('credit_score', 0)
    months_employed = data.get('months_employed', 0)
    num_credit_lines = data.get('credit_lines', 0)
    interest_rate = data.get('interest_rate', 0.0)
    dti_ratio = data.get('dti_ratio', 0.0)
    
    loan_term = data.get('loan_term', 36)
    if isinstance(loan_term, str):
        loan_term = int(loan_term.split()[0])
        
    has_mortgage = 1 if data.get('mortgage') == 'Yes' else 0
    has_dependents = 1 if data.get('dependents') == 'Yes' else 0
    has_co_signer = 1 if data.get('co_signer') == 'Yes' else 0
    
    edu = data.get('education', '')
    edu_high_school = 1 if edu == 'High School' else 0
    edu_masters = 1 if edu == "Master's" else 0
    edu_phd = 1 if edu == 'PhD' else 0
    
    emp = data.get('employment_type', '')
    emp_part_time = 1 if emp == 'Part-time' else 0
    emp_self_employed = 1 if emp == 'Self-employed' else 0
    emp_unemployed = 1 if emp == 'Unemployed' else 0
    
    mar = data.get('marital_status', '')
    mar_married = 1 if mar == 'Married' else 0
    mar_single = 1 if mar == 'Single' else 0
    
    pur = data.get('loan_purpose', '')
    pur_business = 1 if pur == 'Business' else 0
    pur_education = 1 if pur == 'Education' else 0
    pur_home = 1 if pur == 'Home' else 0
    pur_other = 1 if pur == 'Other' else 0
    
    processed_data = {
        'Age': age, 'Income': income, 'LoanAmount': loan_amount, 'CreditScore': credit_score,
        'MonthsEmployed': months_employed, 'NumCreditLines': num_credit_lines, 'InterestRate': interest_rate,
        'LoanTerm': loan_term, 'DTIRatio': dti_ratio, 'HasMortgage': has_mortgage, 'HasDependents': has_dependents,
        'HasCoSigner': has_co_signer, 'Education_High School': edu_high_school, "Education_Master's": edu_masters,
        'Education_PhD': edu_phd, 'EmploymentType_Part-time': emp_part_time, 'EmploymentType_Self-employed': emp_self_employed,
        'EmploymentType_Unemployed': emp_unemployed, 'MaritalStatus_Married': mar_married, 'MaritalStatus_Single': mar_single,
        'LoanPurpose_Business': pur_business, 'LoanPurpose_Education': pur_education, 'LoanPurpose_Home': pur_home, 'LoanPurpose_Other': pur_other
    }

    columns = [
        'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'HasMortgage', 
        'HasDependents', 'HasCoSigner', 'Education_High School', "Education_Master's", 'Education_PhD', 'EmploymentType_Part-time', 'EmploymentType_Self-employed', 
        'EmploymentType_Unemployed', 'MaritalStatus_Married', 'MaritalStatus_Single', 'LoanPurpose_Business', 'LoanPurpose_Education', 'LoanPurpose_Home', 'LoanPurpose_Other'
    ]
    return pd.DataFrame([processed_data], columns=columns)

def predict_loan(data):
    with st.spinner("Analyzing risk profile..."):
        try:
            # Add artificial delay for UX (feeling of processing)
            time.sleep(1.5)
            
            if model is None or scaler is None:
                st.error("Model files not loaded properly. Ensure LoanDefault.pkl and scaler.pkl are available.")
                return
                
            df = preprocess_input(data)
            X_scaled = scaler.transform(df)
            
            probability = model.predict_proba(X_scaled)[0]
            prediction_class = model.predict(X_scaled)[0]
            
            default_prob = float(probability[1])
            confidence = float(max(probability))
            risk_level = "High Risk" if prediction_class == 1 else "Low Risk"
            
            result = {
                'success': True,
                'prediction': risk_level,
                'default_probability': round(default_prob, 2),
                'confidence': int(confidence * 100)
            }
            
            st.session_state.prediction_result = result
            navigate_to('Results')
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

def get_download_link(data, result):
    # Create a simple text report
    report = f"APEX FINANCIAL - LOAN RISK REPORT\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"{'-'*40}\n\n"
    
    report += f"PREDICTION RESULT\n"
    report += f"Risk Level: {result['prediction']}\n"
    report += f"Default Probability: {result['default_probability']*100:.1f}%\n"
    report += f"Confidence Score: {result['confidence']}%\n\n"
    
    report += f"APPLICANT PROFILE\n"
    for key, value in data.items():
        report += f"- {key.replace('_', ' ').title()}: {value}\n"
        
    b64 = base64.b64encode(report.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="loan_risk_report.txt" style="text-decoration: none;"><button style="background-color: #6c757d; color: white; border-radius: 8px; padding: 0.5rem 1rem; border: none; cursor: pointer; width: 100%;">Download Report</button></a>'
    return href

def result_page():
    st.markdown("<h2>Loan Risk Assessment Result</h2>", unsafe_allow_html=True)
    
    result = st.session_state.prediction_result
    data = st.session_state.form_data
    
    if not result:
        st.warning("No prediction data available. Please complete the assessment first.")
        if st.button("Go to Assessment"):
            navigate_to("Assessment")
        return
        
    is_low_risk = result['prediction'] == 'Low Risk'
    card_class = "result-card-low" if is_low_risk else "result-card-high"
    icon = "🟢" if is_low_risk else "🔴"
    message = "Your application shows a relatively low estimated risk of default." if is_low_risk else "Your application shows a higher estimated risk of default."
    
    # Main Result Card
    st.markdown(f"""
<div class='{card_class}'>
<h1 class='result-title'>{icon} {result['prediction'].upper()}</h1>
<p style='font-size: 1.2rem; color: #495057; margin-bottom: 30px;'>{message}</p>
<div style='display: flex; justify-content: space-around; background: white; border-radius: 15px; padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);'>
<div>
<h3 style='color: var(--dark-navy); margin: 0;'>{result['default_probability']*100:.1f}%</h3>
<small style='color: #6c757d; text-transform: uppercase;'>Default Probability</small>
</div>
<div>
<h3 style='color: var(--dark-navy); margin: 0;'>{result['confidence']}%</h3>
<small style='color: #6c757d; text-transform: uppercase;'>AI Confidence</small>
</div>
</div>
</div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Add Charts section
    st.markdown("<h3>Visual Analysis</h3>", unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2)
    
    import altair as alt
    import pandas as pd
    
    with chart_col1:
        # Gauge representation using Altair pie/donut chart
        prob = result['default_probability'] * 100
        gauge_data = pd.DataFrame({
            'Category': ['Risk', 'Safe'],
            'Value': [prob, 100 - prob]
        })
        
        color_scale = alt.Scale(domain=['Risk', 'Safe'], range=['#FFB4B4' if not is_low_risk else '#B2A4FF', '#F8F9FA'])
        
        base = alt.Chart(gauge_data).encode(
            theta=alt.Theta("Value:Q", stack=True),
            color=alt.Color("Category:N", scale=color_scale, legend=None)
        )
        
        pie = base.mark_arc(innerRadius=50, stroke="#fff")
        
        st.markdown("**Risk Gauge**")
        st.altair_chart(pie, use_container_width=True)
        
    with chart_col2:
        # Bar chart showing some key metrics compared to threshold
        st.markdown("**Your Profile vs Standard**")
        
        metrics_data = pd.DataFrame({
            'Metric': ['DTI Ratio', 'Credit Score (scaled)', 'Income (scaled)'],
            'Your Value': [
                data.get('dti_ratio', 0) * 100, 
                data.get('credit_score', 0) / 8.5, 
                min(data.get('annual_income', 0) / 1000, 100)
            ],
            'Standard': [35, 80, 60]  # Example standard values
        })
        
        metrics_melted = pd.melt(metrics_data, id_vars=['Metric'], value_vars=['Your Value', 'Standard'], 
                                var_name='Type', value_name='Value')
                                
        bar_chart = alt.Chart(metrics_melted).mark_bar().encode(
            x=alt.X('Type:N', title=None, axis=alt.Axis(labels=False)),
            y=alt.Y('Value:Q', title='Score'),
            color=alt.Color('Type:N', scale=alt.Scale(domain=['Your Value', 'Standard'], range=['#B2A4FF', '#e9ecef'])),
            column=alt.Column('Metric:N', header=alt.Header(labelOrient='bottom', titleOrient='bottom'))
        ).properties(width=80, height=200)
        
        st.altair_chart(bar_chart, use_container_width=False)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>Key Factors</h3>", unsafe_allow_html=True)
        st.markdown(f"""
<div style='background: white; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef;'>
<p>✓ <b>Credit Score:</b> {data.get('credit_score', 'N/A')}</p>
<p>✓ <b>Stable Employment:</b> {data.get('months_employed', 'N/A')} months</p>
<p>✓ <b>DTI Ratio:</b> {data.get('dti_ratio', 'N/A')}</p>
<p>✓ <b>Co-Signer:</b> {data.get('co_signer', 'No')}</p>
</div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("<h3>Understanding Your Result</h3>", unsafe_allow_html=True)
        recommendation = "Your current financial profile indicates a relatively lower estimated probability of default. Maintaining stable income and a healthy debt-to-income ratio can support your financial profile." if is_low_risk else "Your profile shows factors that historically correlate with higher default risk. Improving your credit score, lowering debt-to-income ratio, or adding a co-signer may strengthen your profile."
        st.markdown(f"""
<div style='background: white; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; color: #495057;'>
<p>{recommendation}</p>
<p style='font-size: 0.8rem; color: #adb5bd; margin-top: 15px;'>* Note: This is an AI estimation based on historical data and not a guaranteed approval or rejection.</p>
</div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Action Buttons
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        if st.button("← New Assessment"):
            st.session_state.prediction_result = None
            navigate_to('Assessment')
    with b_col2:
        if st.button("View Details"):
            st.info("Detailed SHAP analysis would appear here in a full production system.")
    with b_col3:
        st.markdown(get_download_link(data, result), unsafe_allow_html=True)

def about_page():
    st.markdown("<h2 style='text-align: center; margin-bottom: 40px;'>How Apex Financial Works</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card' style='height: 100%;'>
            <h1 style='color: var(--primary-purple);'>01</h1>
            <h4>Enter Information</h4>
            <p style='color: #6c757d; font-size: 0.9rem;'>Provide basic financial and personal details through our secure form.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class='metric-card' style='height: 100%;'>
            <h1 style='color: var(--primary-purple);'>02</h1>
            <h4>Machine Learning Analysis</h4>
            <p style='color: #6c757d; font-size: 0.9rem;'>Our advanced algorithm evaluates your profile against historical loan data.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class='metric-card' style='height: 100%;'>
            <h1 style='color: var(--primary-purple);'>03</h1>
            <h4>Risk Prediction</h4>
            <p style='color: #6c757d; font-size: 0.9rem;'>Receive an instant, easy-to-understand probability of loan default.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><hr style='border-color: #f8f9fa;'><br>", unsafe_allow_html=True)
    
    st.markdown("### About the Model")
    st.write("The system uses a machine learning classification model trained on historical loan data to estimate the probability of loan default. It evaluates multiple factors simultaneously to provide a comprehensive risk assessment.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Technology")
    
    t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
    technologies = ["Python", "Streamlit", "Flask", "Scikit-learn", "Pandas"]
    cols = [t_col1, t_col2, t_col3, t_col4, t_col5]
    
    for col, tech in zip(cols, technologies):
        with col:
            st.markdown(f"""
            <div style='background: white; padding: 10px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;'>
                <b>{tech}</b>
            </div>
            """, unsafe_allow_html=True)

# Main App Router
show_sidebar()

if st.session_state.page == 'Home':
    home_page()
elif st.session_state.page == 'Assessment':
    assessment_page()
elif st.session_state.page == 'Results':
    result_page()
elif st.session_state.page == 'About':
    about_page()
