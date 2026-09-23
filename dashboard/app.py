import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Quantum Image Processing",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("Quantum Image Processing Framework")

st.subheader("MRI Image Enhancement & Comparison")

st.write(
    "Compare original MRI images with quantum-inspired "
    "processed images."
)


# ==========================================
# METRIC FUNCTION
# ==========================================

def calculate_metrics(original, processed):

    # Mean Squared Error
    mse = np.mean(
        (original.astype(float) - processed.astype(float)) ** 2
    )

    # Peak Signal-to-Noise Ratio
    psnr = cv2.PSNR(
        original,
        processed
    )

    # Structural Similarity
    ssim_value = ssim(
        original,
        processed,
        data_range=255
    )

    return mse, psnr, ssim_value


# ==========================================
# SELECT MRI IMAGE
# ==========================================

st.subheader("Select MRI Image")

selected = st.selectbox(
    "Choose an image:",
    ["brain1", "brain2", "brain3"]
)


# ==========================================
# IMAGE PATHS
# ==========================================

original_path = (
    f"dataset/images/{selected}.jpeg"
)

processed_path = (
    f"dataset/processed/{selected}_quantum.png"
)


# ==========================================
# READ IMAGES
# ==========================================

original = cv2.imread(
    original_path,
    cv2.IMREAD_GRAYSCALE
)

processed = cv2.imread(
    processed_path,
    cv2.IMREAD_GRAYSCALE
)


# ==========================================
# CHECK IMAGES
# ==========================================

if original is None:

    st.error(
        f"Original image not found: {original_path}"
    )

    st.stop()


if processed is None:

    st.error(
        f"Processed image not found: {processed_path}"
    )

    st.stop()


# ==========================================
# CALCULATE METRICS
# ==========================================

mse, psnr, ssim_value = calculate_metrics(
    original,
    processed
)


# ==========================================
# DISPLAY IMAGES
# ==========================================

st.subheader("Image Comparison")


col1, col2 = st.columns(2)


with col1:

    st.write("### Original MRI")

    st.image(
        original,
        caption=f"{selected} - Original",
        width=300
    )


with col2:

    st.write("### Quantum-Inspired MRI")

    st.image(
        processed,
        caption=f"{selected} - Quantum-Inspired",
        width=300
    )


# ==========================================
# COMPARISON RESULTS
# ==========================================

st.subheader("Comparison Results")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "MSE",
        f"{mse:.4f}"
    )


with col2:

    st.metric(
        "PSNR",
        f"{psnr:.4f} dB"
    )


with col3:

    st.metric(
        "SSIM",
        f"{ssim_value:.4f}"
    )


# ==========================================
# METRIC EXPLANATION
# ==========================================

st.subheader("Metric Explanation")


st.write(
    "**MSE:** Measures the pixel-level difference "
    "between the original and processed image. "
    "Lower means more similar."
)


st.write(
    "**PSNR:** Measures the similarity/quality "
    "relative to the original image. "
    "Higher means more similar."
)


st.write(
    "**SSIM:** Measures structural similarity "
    "between the two images. "
    "A value closer to 1 means greater similarity."
)


# ==========================================
# PROJECT NOTE
# ==========================================

st.warning(
    "Note: This project performs image enhancement "
    "and comparison. It does not automatically "
    "diagnose cancer."
)