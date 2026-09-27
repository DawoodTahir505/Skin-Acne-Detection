import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

# Page config
st.set_page_config(page_title="Skin Acne Detector", layout="wide")
st.title("🔍 Skin Acne Detector")

# Load model
@st.cache_resource
def load_model():
    model = YOLO("/mnt/user-data/uploads/skin-acne-detector.pt")
    return model

model = load_model()

# Sidebar - Confidence threshold
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# Upload image
uploaded_file = st.file_uploader("Upload skin image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    # Run inference
    with st.spinner("Detecting acne..."):
        # Save temp image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name
        
        # Predict
        results = model.predict(temp_path, conf=conf_threshold, imgsz=640)
        
        # Plot results
        result = results[0]
        annotated_image = result.plot()
        annotated_pil = Image.fromarray(annotated_image[:, :, ::-1])
        
        # Cleanup
        os.unlink(temp_path)
    
    with col2:
        st.subheader("Detection Results")
        st.image(annotated_pil, use_column_width=True)
    
    # Display statistics
    detections = result.boxes
    st.subheader("Detection Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Detections", len(detections))
    
    with col2:
        if len(detections) > 0:
            avg_conf = detections.conf.mean().item()
            st.metric("Avg Confidence", f"{avg_conf:.2%}")
        else:
            st.metric("Avg Confidence", "N/A")
    
    with col3:
        if len(detections) > 0:
            max_conf = detections.conf.max().item()
            st.metric("Max Confidence", f"{max_conf:.2%}")
        else:
            st.metric("Max Confidence", "N/A")
    
    # Detailed results
    if len(detections) > 0:
        st.subheader("Detailed Detections")
        det_data = []
        for i, box in enumerate(detections):
            det_data.append({
                "Acne #": i + 1,
                "Confidence": f"{box.conf.item():.2%}",
                "X1": f"{int(box.xyxy[0][0].item())}",
                "Y1": f"{int(box.xyxy[0][1].item())}",
                "X2": f"{int(box.xyxy[0][2].item())}",
                "Y2": f"{int(box.xyxy[0][3].item())}"
            })
        st.dataframe(det_data, use_container_width=True)
    else:
        st.info("No acne detected in the image.")
