```python
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
        st.image(
            image,
            use_container_width=True
        )

    # Detection
    with st.spinner("Detecting acne..."):

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as tmp:

                image.save(tmp.name)
                temp_path = tmp.name

            # Fixed confidence threshold
            results = model.predict(
                temp_path,
                conf=0.25,
                imgsz=640
            )

            result = results[0]

            # Annotated image
            annotated_image = result.plot()

            # Convert BGR to RGB
            annotated_pil = Image.fromarray(
                annotated_image[:, :, ::-1]
            )

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    # Detection result
    with col2:
        st.subheader("Detection Result")
        st.image(
            annotated_pil,
            use_container_width=True
        )

    # No detection message
    if len(result.boxes) == 0:
        st.info("No acne detected.")
```
