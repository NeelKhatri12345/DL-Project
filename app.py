
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import layers, Model

IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)

    # Build model by passing a dummy input first
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model(dummy, training=False)

    # Load weights saved as numpy
    weights = np.load("deepfake_weights.npy", allow_pickle=True)
    model.set_weights(weights)
    return model

model = load_model()

st.title("Deepfake Face Detector")
st.write("Upload a face image to check if it is **real** or **AI-generated (fake)**.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    with st.spinner("Analysing..."):
        prob = float(model.predict(arr, verbose=0)[0][0])

    # fake=0, real=1 -> prob > 0.5 means real
    label = "REAL" if prob > 0.5 else "FAKE"
    confidence = prob if prob > 0.5 else 1 - prob

    col1, col2 = st.columns(2)
    col1.metric("Prediction", label)
    col2.metric("Confidence", f"{confidence*100:.1f}%")
    st.progress(float(confidence))
