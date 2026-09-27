import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Skin Acne Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Skin Acne Detector")
st.write("Upload a skin image to detect acne.")

# Load model
@st.cache_resource
def load_model():
    return YOLO("skin-acne-detector.pt")

model = load_model()

# Sidebar
st.sidebar.header("Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0,
    1.0,
    0.25,
    0.05
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload skin image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # Original image
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # Detection
    with st.spinner("Detecting acne..."):

        temp_path = None

        try:
            # Create temporary image
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as tmp:

                image.save(tmp.name)
                temp_path = tmp.name

            # Prediction
            results = model.predict(
                temp_path,
                conf=conf_threshold,
                imgsz=640
            )

            result = results[0]

            # Annotated image
            annotated_image = result.plot()

            # BGR → RGB
            annotated_pil = Image.fromarray(
                annotated_image[:, :, ::-1]
            )

        finally:
            # Remove temporary file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    # Results
    with col2:
        st.subheader("Detection Results")
        st.image(
            annotated_pil,
            use_container_width=True
        )

    # Detection statistics
    detections = result.boxes

    st.subheader("Detection Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Detections",
            len(detections)
        )

    with col2:
        if len(detections) > 0:
            avg_conf = detections.conf.mean().item()

            st.metric(
                "Avg Confidence",
                f"{avg_conf:.2%}"
            )
        else:
            st.metric(
                "Avg Confidence",
                "N/A"
            )

    with col3:
        if len(detections) > 0:
            max_conf = detections.conf.max().item()

            st.metric(
                "Max Confidence",
                f"{max_conf:.2%}"
            )
        else:
            st.metric(
                "Max Confidence",
                "N/A"
            )

    # Detailed detections
    if len(detections) > 0:

        st.subheader("Detailed Detections")

        det_data = []

        for i, box in enumerate(detections):

            det_data.append({
                "Acne #": i + 1,
                "Confidence": f"{box.conf.item():.2%}",
                "X1": int(box.xyxy[0][0].item()),
                "Y1": int(box.xyxy[0][1].item()),
                "X2": int(box.xyxy[0][2].item()),
                "Y2": int(box.xyxy[0][3].item())
            })

        st.dataframe(
            det_data,
            use_container_width=True
        )

    else:
        st.info(
            "No acne detected. Try lowering the confidence threshold."
        )
