import streamlit as st
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import tensorflow as tf

# --- 1. Page Setup ---
st.set_page_config(
    page_title="Teeth Disease Classifier",
    page_icon="🦷",
    layout="centered"
)

# --- 2. Class Labels ---
diseases = [
    'Canker Sores (CaS)',
    'Cold Sores (CoS)',
    'Gum Disease (Gum)',
    'Mouth Cancer (MC)',
    'Oral Cancer (OC)',
    'Oral Lichen Planus (OLP)',
    'Oral Thrush (OT)'
]
import streamlit as st

st.title("🦷 Teeth Disease Classifier 🦷")

# Inject CSS directly
st.markdown("""
<style>
            
h1 {
    font-size: 50px !important;  
    color: #1abc9c;
    text-align: center;
}
h2 {
    font-size: 30px !important; 
    color: #3498db;
}
.big-text {
    font-size: 28px !important;  
    color: black;
}
.prediction {
    font-size: 30px !important;   
    color: red;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<h2 class="big-text">Upload an image to classify oral diseases</h2>', unsafe_allow_html=True)
st.markdown('<p class="prediction">Predicted Disease:</p>', unsafe_allow_html=True)

# --- 3. Load Model (cached so it doesn't reload every run) ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\Mariam\Downloads\Model_Teeth_disease_classification.h5")
    return model

model = load_model()

# --- 4. Upload Image ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # --- 5. Preprocess Image ---
    img_array = np.array(image.resize((256, 256))) / 255.0  # normalize 0-1
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    # --- 6. Make Prediction ---
    with st.spinner('Classifying...'):
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100

    # --- 7. Show Results ---
    st.success(f"Predicted Disease: **{diseases[class_idx]}**")
    st.info(f"Confidence: {confidence:.2f}%")
