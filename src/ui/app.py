import streamlit as st
import requests
from PIL import Image
import io

# --- CONFIG ---
API_URL = "http://localhost:8000"  # Change if deploying elsewhere

st.set_page_config(page_title="Parkinson's Detection", layout="centered")

# --- Centered Header ---
st.markdown("""
    <h1 style='text-align:center;'>Parkinson's Detection</h1>
""", unsafe_allow_html=True)

# --- Fetch available models and metrics ---
@st.cache_data(show_spinner=False)
def get_models_and_metrics():
    try:
        resp = requests.get(f"{API_URL}/models")
        resp.raise_for_status()
        data = resp.json()
        return data["models"]
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")
        return {}

models = get_models_and_metrics()
model_names = list(models.keys())

# --- Expander with About and Steps ---
with st.expander("About", expanded=True):
    st.markdown("""
    <div>
        This application helps in the early detection of Parkinson's disease by analyzing hand-drawn images (currently trained with spirals and waves). 
        Users can select a trained model, upload their drawing, and receive a prediction indicating whether the drawing shows signs of Parkinson's or is healthy.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    **How to use:**
    1. **Select Model**
    2. **Upload Image**
    3. **Click Predict**
    """)

# --- Main UI: Model selector, metrics, image uploader, image preview ---
col_model, col_upload = st.columns([2, 2])
with col_model:
    selected_model = st.selectbox("Select Model", model_names)
    metrics = models[selected_model]
    mcol1, mcol2 = st.columns(2)
    mcol1.markdown(f"<span style='font-size:1rem;'>Accuracy</span><br><span style='font-size:1.5rem;font-weight:bold'>{metrics['accuracy']:.2%}</span>", unsafe_allow_html=True)
    mcol1.markdown(f"<span style='font-size:1rem;'>Precision</span><br><span style='font-size:1.5rem;font-weight:bold'>{metrics['precision']:.2%}</span>", unsafe_allow_html=True)
    mcol2.markdown(f"<span style='font-size:1rem;'>Recall</span><br><span style='font-size:1.5rem;font-weight:bold'>{metrics['recall']:.2%}</span>", unsafe_allow_html=True)
    if metrics.get('f1_score'):
        mcol2.markdown(f"<span style='font-size:1rem;'>F1 Score</span><br><span style='font-size:1.5rem;font-weight:bold'>{metrics['f1_score']:.2%}</span>", unsafe_allow_html=True)
with col_upload:
    image_file = st.file_uploader("Upload Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if image_file:
        st.image(image_file, caption="Uploaded Image", width=150)

# --- Predict button and prediction below ---
if 'image_file' in locals() and image_file:
    predict_clicked = st.button("Predict")
    if predict_clicked:
        with st.spinner("Predicting..."):
            try:
                image_file.seek(0)
                files = {"file": (image_file.name, image_file, image_file.type)}
                data = {"model_name": selected_model}
                resp = requests.post(f"{API_URL}/predict", files=files, data=data)
                try:
                    resp.raise_for_status()
                except Exception:
                    st.error(f"Backend error: {resp.text}")
                    raise
                result = resp.json()
                pred = result['prediction']
                color = "#27ae60" if pred.lower() == "healthy" else "#e74c3c"
                st.markdown(f"<div style='margin-top:1em;font-size:1.3rem;font-weight:bold;'>Prediction: <span style='color:{color};'>{pred}</span></div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
