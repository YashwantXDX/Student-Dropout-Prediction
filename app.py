import streamlit as st
import joblib
import numpy as np
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Student Drop Prediction Model",
    page_icon="🎓",
    layout="wide",
)

# --- PREMIUM CUSTOM CSS ---
st.markdown("""
    <style>
    /* Overall Background - Deep Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }

    /* Heading Styling - Bright & Bold */
    .main-title {
        font-size: 50px !important;
        font-weight: 900;
        background: -webkit-linear-gradient(#38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .sub-title {
        color: #94a3b8;
        text-align: center;
        font-size: 18px;
        margin-bottom: 40px;
    }

    /* Section Headings */
    h3 {
        color: #fbbf24 !important; /* Gold color for visibility */
        font-weight: 700 !important;
        border-left: 5px solid #38bdf8;
        padding-left: 10px;
    }

    /* Card Effect for Inputs */
    [data-testid="stVerticalBlock"] > div:has(div.stNumberInput) {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 15px 30px;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }

    /* Result Box */
    .prediction-card {
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin-top: 30px;
        border: 2px solid;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING (Cached for performance) ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("student_dropout_model.pkl")
        features = joblib.load("model_features.pkl")
        return model, features
    except:
        return None, None

model, features = load_assets()

# --- HEADER ---
st.markdown('<p class="main-title">🎓 STUDENT DROPOUT PREDICTION</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predictive Intelligence for Academic Institutions</p>', unsafe_allow_html=True)

# --- INPUT SECTION ---
application_mode_mapping = {
    "1st phase - general contingent": 1, "Ordinance No. 612/93": 2, "1st phase - special contingent (Azores Island)": 5,
    "Holders of other higher courses": 7, "Ordinance No. 854-B/99": 10, "International student (bachelor)": 15,
    "1st phase - special contingent (Madeira Island)": 16, "2nd phase - general contingent": 17,
    "3rd phase - general contingent": 18, "Over 23 years old": 39, "Transfer": 42, "Change of course": 43
}

# Dividing inputs into 2 main containers for better hierarchy
with st.container():
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("📋 Demographic & Admission")
        age = st.number_input("Age at Enrollment", 15, 70, 20)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "👩 Female" if x == 0 else "👨 Male")
        mode_label = st.selectbox("Application Mode", list(application_mode_mapping.keys()))
        app_mode = application_mode_mapping[mode_label]
        admission_grade = st.number_input("Admission Grade (0-200)", 0.0, 200.0, 120.0)

    with col2:
        st.subheader("💰 Financial & Support")
        scholarship = st.selectbox("Scholarship Holder", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes ✅")
        tuition = st.selectbox("Tuition Fees Up to Date", [0, 1], format_func=lambda x: "No ❌" if x == 0 else "Yes ✅")
        debtor = st.selectbox("Debtor Status", [0, 1], format_func=lambda x: "No Debt" if x == 0 else "Has Debt ⚠️")
        prev_grade = st.number_input("Previous Qualification Grade", 0.0, 200.0, 120.0)

st.markdown("<br>", unsafe_allow_html=True)

with st.container():
    st.subheader("📝 Academic Performance (Current Year)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        u1_app = st.number_input("Sem 1 Approved Units", 0, 20, 5)
    with c2:
        u1_grd = st.number_input("Sem 1 Average Grade", 0.0, 20.0, 12.0)
    with c3:
        u2_app = st.number_input("Sem 2 Approved Units", 0, 20, 5)
    with c4:
        u2_grd = st.number_input("Sem 2 Average Grade", 0.0, 20.0, 12.0)

st.markdown("<br>", unsafe_allow_html=True)

# --- PREDICTION ---
if st.button("🔍 ANALYZE STUDENT DATA", use_container_width=True):
    if model:
        with st.spinner('Computing Probability...'):
            time.sleep(1) # For dramatic effect
            
            input_data = np.array([[
                age, gender, app_mode, scholarship, admission_grade,
                prev_grade, u1_app, u1_grd, u2_app, u2_grd, tuition, debtor
            ]])

            prediction = model.predict(input_data)[0]
            probs = model.predict_proba(input_data)[0]
            confidence = probs[prediction] * 100

            st.markdown("---")
            
            if prediction == 1:
                st.markdown(f"""
                    <div class="prediction-card" style="background: rgba(16, 185, 129, 0.1); border-color: #10b981; color: #10b981;">
                        <h1 style='margin:0;'>🏆 SUCCESS: LIKELY TO GRADUATE</h1>
                        <p style='font-size: 20px;'>The model is <b>{confidence:.2f}%</b> confident in this student's progress.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-card" style="background: rgba(239, 68, 68, 0.1); border-color: #ef4444; color: #ef4444;">
                        <h1 style='margin:0;'>⚠️ ALERT: DROPOUT RISK</h1>
                        <p style='font-size: 20px;'>The model is <b>{confidence:.2f}%</b> confident that the student may face difficulties.</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.error("Model files not found. Please check your .pkl files.")

# --- FOOTER ---
st.markdown("<br><p style='text-align: center; color: #475569;'>Made by Yashwant & Team • 2026</p>", unsafe_allow_html=True)