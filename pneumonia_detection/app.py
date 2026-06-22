import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dense as TFDense
from PIL import Image

class Dense(TFDense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

@st.cache_resource
def load_pneumonia_model():
    return tf.keras.models.load_model(
        "pneumonia_model.h5",
        custom_objects={"Dense": Dense},
        compile=False
    )

model = load_pneumonia_model()

def preprocess_image(image: Image.Image):
    img = image.convert("L")
    img = np.array(img)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = img.reshape(1, 150, 150, 1)
    return img

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #040d18;
    }

    .main .block-container {
        padding: 2rem 2.5rem 3rem;
        max-width: 1200px;
    }

    /* ── Header ── */
    .header-wrap {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 2.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .header-icon {
        font-size: 4.5rem;
        line-height: 1;
    }
    .header-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #f0f6ff;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .header-sub {
        font-size: 2rem;
        color: #5a7a9a;
        margin: 2px 0 0;
        font-weight: 400;
    }

    /* ── Panel cards ── */
    .panel {
        background: #07111f;
        border: 1px solid #0e2035;
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
    }
    .panel-label {
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #2e5f8a;
        margin-bottom: 1rem;
    }

    /* ── Upload zone ── */
    div[data-testid="stFileUploader"] {
        background: #040d18;
        border: 1.5px dashed #0e2a45;
        border-radius: 12px;
        padding: 8px;
        transition: border-color 0.2s;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #1b5fa8;
    }
    div[data-testid="stFileUploader"] label {
        color: #3d7ab5 !important;
        font-size: 2rem !important;
    }

    /* ── Predict button ── */
    .stButton > button {
        width: 100%;
        background: #0f4c8a;
        color: #d6eaff;
        border: 2px solid #1a6ab5;
        border-radius: 10px;
        padding: 2rem 3rem;
        font-size: 6rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        margin-top: 16px;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: #1a6ab5;
        border-color: #2980d4;
        color: #ffffff;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(26, 106, 181, 0.3);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Result badges ── */
    .result-badge {
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .result-badge.normal {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .result-badge.pneumonia {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    .result-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .dot-normal { background: #10b981; box-shadow: 0 0 8px #10b981; }
    .dot-pneumonia { background: #ef4444; box-shadow: 0 0 8px #ef4444; }
    .result-label {
        font-size: 4rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .label-normal { color: #34d399; }
    .label-pneumonia { color: #f87171; }
    .result-conf {
        font-size: 1.25rem;
        color: #4a6f8a;
        margin-top: 2px;
    }

    /* ── Confidence bar ── */
    .conf-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 1rem 0 0.4rem;
    }
    .conf-label { font-size: 1.5rem; color: #2e5f8a; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; }
    .conf-value { font-size: 2rem; color: #d6eaff; font-weight: 600; }
    .conf-track {
        width: 100%;
        height: 6px;
        background: #0a1e33;
        border-radius: 99px;
        overflow: hidden;
    }
    .conf-fill-normal { height: 100%; background: linear-gradient(90deg, #059669, #34d399); border-radius: 99px; }
    .conf-fill-pneumonia { height: 100%; background: linear-gradient(90deg, #b91c1c, #f87171); border-radius: 99px; }

    /* ── Raw output ── */
    .raw-output {
        background: #030a13;
        border: 1px solid #0a1e33;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 1rem;
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 2rem;
        color: #3d6e96;
    }
    .raw-key { color: #2e5f8a; }
    .raw-val { color: #7fb3d3; }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #1c3d5a;
    }
    .empty-icon { font-size: 2.5rem; margin-bottom: 0.75rem; opacity: 0.4; }
    .empty-text { font-size: 0.85rem; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #040d18;
        border-right: 1px solid #0a1e33;
    }
    section[data-testid="stSidebar"] * {
        color: #4a7a9b !important;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #2e6a9e !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }

    /* ── Footer ── */
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #0a1e33;
        text-align: center;
        font-size: 2rem;
        color: #1c3d5a;
        letter-spacing: 0.04em;
    }

    /* hide streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
    <div class="header-icon">🫁</div>
    <div>
        <div class="header-title">Pneumonia Detection</div>
        <div class="header-sub">AI-powered chest X-ray screening · Deep Learning</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## About")
    st.write("Upload a chest X-ray and the model will classify it as **Pneumonia** or **Normal** with a confidence score.")
    st.markdown("### How it works")
    st.write("1. Upload an X-ray image (JPG / PNG)")
    st.write("2. Click **Predict**")
    st.write("3. View result and confidence")
    st.markdown("### Disclaimer")
    st.write("This tool is for **screening only** and does not replace clinical diagnosis.")

# ── Layout ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.1], gap="large")

with col1:
    st.markdown('<div class="panel-label">Upload X-Ray</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Chest X-Ray", use_container_width=True)
        predict_clicked = st.button("Run Analysis →")
    else:
        st.markdown("""
        <div style="border: 1.5px dashed #0e2a45; border-radius: 12px; padding: 2.5rem 1rem; text-align:center; color:#1c3d5a; font-size:1.5rem;">
            <div style="font-size:2rem; margin-bottom:0.5rem; opacity:0.3;">📂</div>
            Drop a chest X-ray here<br>or click to browse
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel-label">Analysis Result</div>', unsafe_allow_html=True)

    if uploaded_file is not None and 'predict_clicked' in locals() and predict_clicked:
        with st.spinner("Analysing..."):
            try:
                img = preprocess_image(image)
                prediction = model.predict(img, verbose=0)[0][0]

                if prediction > 0.5:
                    confidence = prediction * 100
                    badge_class = "pneumonia"
                    dot_class = "dot-pneumonia"
                    label_class = "label-pneumonia"
                    label_text = "Pneumonia Detected"
                    fill_class = "conf-fill-pneumonia"
                else:
                    confidence = (1 - prediction) * 100
                    badge_class = "normal"
                    dot_class = "dot-normal"
                    label_class = "label-normal"
                    label_text = "Normal"
                    fill_class = "conf-fill-normal"

                st.markdown(f"""
                <div class="result-badge {badge_class}">
                    <div class="result-dot {dot_class}"></div>
                    <div>
                        <div class="result-label {label_class}">{label_text}</div>
                        <div class="result-conf">Model confidence score</div>
                    </div>
                </div>

                <div class="conf-row">
                    <span class="conf-label">Confidence</span>
                    <span class="conf-value">{confidence:.1f}%</span>
                </div>
                <div class="conf-track">
                    <div class="{fill_class}" style="width:{confidence:.1f}%"></div>
                </div>

                <div class="raw-output">
                    <span class="raw-key">raw_output</span> &nbsp;·&nbsp; <span class="raw-val">{prediction:.6f}</span><br>
                    <span class="raw-key">threshold</span> &nbsp;&nbsp;·&nbsp; <span class="raw-val">0.500000</span>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📊</div>
            <div class="empty-text">Upload an X-ray and run analysis<br>to see the result here.</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    For screening purposes only &nbsp;·&nbsp; Not a medical diagnosis &nbsp;·&nbsp; Built with TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)