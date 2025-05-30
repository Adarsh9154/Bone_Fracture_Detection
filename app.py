import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("bone_fracture_detection_model.h5")

st.title("Bone Fracture Detection from X-ray")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((224, 224))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    prediction = model.predict(image_array)[0][0]
    result = "Fractured" if prediction > 0.5 else "Normal"

    st.markdown(f"### Prediction: `{result}`")
    st.progress(min(int(prediction * 100), 100))
