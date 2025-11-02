import streamlit as st
import numpy as np
import joblib

# ----------------------- PAGE CONFIG -----------------------
st.set_page_config(
    page_title="Breast Cancer Detection 💗",
    page_icon="💗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ----------------------- LOAD MODEL ------------------------
model = joblib.load("breast_cancer_model.pkl")

# ----------------------- CUSTOM CSS ------------------------
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
        }
        .stApp {
            background-color: #0E1117;
            color: #FFFFFF;
        }
        h1, h2, h3 {
            color: #FF69B4;
            text-align: center;
        }
        .stButton>button {
            background-color: #FF69B4;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1.2em;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #ff4fa1;
            color: #fff;
        }
        .footer {
            text-align: center;
            font-size: 13px;
            margin-top: 2em;
            color: #aaaaaa;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------- HEADER ----------------------------
st.title("💗 Breast Cancer Detection using ML")
st.markdown(
    "<h5 style='text-align:center;'>Predict whether a breast tumor is <span style='color:#00FA9A;'>Benign</span> or <span style='color:#FF6347;'>Malignant</span>.</h5>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ----------------------- SIDEBAR ---------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/5/50/Ribbon_pink.svg",
    width=100,
)
st.sidebar.header("🔍 Input Cell Nucleus Features")

# All features used by model
features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# ----------------------- INPUT FORM ------------------------
st.subheader("🧬 Enter Input Features Below:")
cols = st.columns(3)
input_data = []

for i, feature in enumerate(features):
    with cols[i % 3]:
        val = st.number_input(f"{feature}", min_value=0.0, format="%.3f")
        input_data.append(val)

st.markdown("---")

# ----------------------- PREDICTION ------------------------
if st.button("🩺 Predict"):
    x = np.array(input_data).reshape(1, -1)
    pred = model.predict(x)[0]
    prob = model.predict_proba(x)[0][pred]

    st.markdown("## 🔎 Prediction Result:")
    if pred == 1:
        st.success("✅ *Result:* The tumor is *Benign (Non-Cancerous)* 💚")
    else:
        st.error("⚠ *Result:* The tumor is *Malignant (Cancerous)* ❤")

    st.progress(int(prob * 100))
    st.info(f"*Model Confidence:* {prob*100:.2f}%")

st.markdown("---")

# ----------------------- FOOTER ----------------------------
st.markdown(
    "<div class='footer'>Developed by <b>Rishu Raj, Harsh Pathak, Vansh Gautam  </b> | Dataset: Wisconsin Breast Cancer Diagnostic | Model: Random Forest</div>",
    unsafe_allow_html=True,
)
