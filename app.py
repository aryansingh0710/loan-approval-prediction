import streamlit as st
import pickle
import numpy as np
import pandas as pd
from fpdf2  import FPDF
import plotly.graph_objects as go

# ======================
# Page Config
# ======================
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# ======================
# Load Model
# ======================
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ======================
# CSS
# ======================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(-45deg, #1f4037, #99f2c8, #3a7bd5, #00d2ff);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
    margin-bottom: 20px;
}

.card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(15px);
    padding: 30px;
    border-radius: 20px;
    max-width: 750px;
    margin: auto;
    box-shadow: 0px 8px 32px rgba(0,0,0,0.3);
}

label {
    color: white !important;
    font-weight: 600;
}

.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    height: 3em;
    width: 100%;
}

.risk-badge {
    padding: 10px 20px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
    margin-top: 10px;
    color: white;
    font-size: 18px;
}

.footer-container {
    text-align: center;
    margin-top: 40px;
}

.social-btn {
    display: inline-block;
    padding: 10px 18px;
    margin: 8px;
    border-radius: 8px;
    text-decoration: none;
    font-weight: 600;
    color: white !important;
}

.github { background: #24292e; }
.linkedin { background: #0077b5; }

</style>
""", unsafe_allow_html=True)

# ======================
# Title
# ======================
st.markdown(
    '<p class="title">🏦 Loan Approval Prediction System</p>',
    unsafe_allow_html=True
)

# ======================
# Form
# ======================
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    married = st.selectbox("💍 Married", ["Yes", "No"])
    dependents = st.selectbox("👨‍👩‍👧 Dependents", [0,1,2,3])
    education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])

with col2:
    self_employed = st.selectbox("💼 Self Employed", ["Yes", "No"])
    
    applicant_income = st.number_input(
        "💰 Applicant Income (Yearly ₹)",
        min_value=0,
        max_value=7000000,
        step=50000
    )

    loan_amount = st.number_input(
        "🏦 Loan Amount (₹)",
        min_value=0,
        max_value=10000000,
        step=50000
    )

    credit_history = st.selectbox("📊 Credit History", [1,0])

predict = st.button("🚀 Predict Loan Status")

st.markdown('</div>', unsafe_allow_html=True)

# ======================
# Preprocessing (Log Features)
# ======================
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

total_income = applicant_income

ApplicantIncomelog = np.log(applicant_income + 1)
LoanAmountlog = np.log(loan_amount + 1)
Total_Income_log = np.log(total_income + 1)
Loan_Amount_Term_log = np.log(360 + 1)

input_dict = {
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'Credit_History': credit_history,
    'Property_Area': 0,
    'ApplicantIncomelog': ApplicantIncomelog,
    'LoanAmountlog': LoanAmountlog,
    'Loan_Amount_Term_log': Loan_Amount_Term_log,
    'Total_Income_log': Total_Income_log
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

features = input_df.values

# ======================
# Gauge
# ======================
def gauge_chart(prob):
    value = prob * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': "%"},
        title={'text': "Approval Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00c6ff"},
            'steps': [
                {'range': [0, 40], 'color': "#ff4b4b"},
                {'range': [40, 70], 'color': "#ffa500"},
                {'range': [70, 100], 'color': "#00ff88"}
            ]
        }
    ))
    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}
    )
    return fig

# ======================
# Risk Function
# ======================
def get_risk_level(prob):
    if prob >= 0.7:
        return "Low Risk", "#00ff88"
    elif prob >= 0.4:
        return "Medium Risk", "#ffa500"
    else:
        return "High Risk", "#ff4b4b"

# ======================
# PDF
# ======================
def create_pdf(result, prob):
    clean_result = result.replace("✅", "").replace("❌", "")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    pdf.cell(200, 10, txt="Loan Prediction Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Result: {clean_result}", ln=True)
    pdf.cell(200, 10, txt=f"Probability: {prob*100:.2f}%", ln=True)

    file_path = "prediction_report.pdf"
    pdf.output(file_path)
    return file_path

# ======================
# Prediction
# ======================
if predict:

    prediction = model.predict(features)[0]

    try:
        prob = model.predict_proba(features)[0][1]
    except:
        prob = 0.5

    st.plotly_chart(gauge_chart(prob), use_container_width=True)

    # Risk Badge
    risk_label, risk_color = get_risk_level(prob)

    st.markdown(
        f'<div class="risk-badge" style="background:{risk_color};">'
        f'Risk Level: {risk_label}'
        '</div>',
        unsafe_allow_html=True
    )

    if prediction == 1:
        result_text = "Loan Approved ✅"
        st.success(result_text)
        st.balloons()
    else:
        result_text = "Loan Rejected ❌"
        st.error(result_text)
        st.snow()

    pdf_file = create_pdf(result_text, prob)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄 Download Prediction Report",
            data=f,
            file_name="Loan_Report.pdf",
            mime="application/pdf"
        )

# ======================
# Footer
# ======================
st.markdown("""
<div class="footer-container">

<p style="color:white;font-size:18px;">
👨‍💻 Developed by Aryan Singh
</p>

<a class="social-btn github"
href="https://github.com/aryansingh0710"
target="_blank">
🐙 GitHub
</a>

<a class="social-btn linkedin"
href="https://www.linkedin.com/in/aryan-singh-ba6000252"
target="_blank">
💼 LinkedIn
</a>



</div>
""", unsafe_allow_html=True)