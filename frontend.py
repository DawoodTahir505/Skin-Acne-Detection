import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Acne Detection App", layout="centered")

st.title("Acne Detection with YOLO11")
st.write("Upload an image to detect features trained on the acne dataset.")

@st.cache_resource
def load_model():
    # Load the best weights generated during your training run
    model_path = r"C:\Users\Pc\Desktop\YOLO-Project\runs\detect\train-9\weights\best.pt"
    return YOLO(model_path)

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Run Detection"):
        with st.spinner("Detecting..."):
            # Run inference on the uploaded image
            results = model(image)
            
            # The plot() method returns a numpy array in BGR format
            annotated_bgr = results[0].plot()
            
            # Convert BGR to RGB for correct color rendering in Streamlit
            annotated_rgb = annotated_bgr[:, :, ::-1]
            
            st.image(annotated_rgb, caption="Detection Results", use_container_width=True)