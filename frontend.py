import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Configure page UI
st.set_page_config(page_title="Skin Detection App", page_icon="🩺", layout="centered")

st.title("🩺 Skin Detection App")
st.write("Upload a skin image to detect acne and skin conditions using your trained YOLO model.")

# Relative path to your model file inside your project folder
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error(f"Error: Model file `{MODEL_PATH}` was not found. Please upload `best.pt` into your repository root folder.")
else:
    # Sidebar control for confidence threshold
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

    # Image upload widget
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Run Skin Detection"):
            with st.spinner("Analyzing image..."):
                # Run YOLO inference
                results = model.predict(image, conf=conf_threshold)
                
                # Render detected bounding boxes (Convert BGR output to RGB for display)
                annotated_bgr = results[0].plot()
                annotated_rgb = annotated_bgr[:, :, ::-1]
                
                st.image(annotated_rgb, caption="Detection Results", use_container_width=True)
                st.success("Detection completed!")
