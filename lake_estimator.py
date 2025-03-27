import cv2
import numpy as np
import streamlit as st
from PIL import Image

def process_image(image):
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        st.error("No lake detected! Try adjusting the HSV values.")
        return None, None, None

    lake_contour = max(contours, key=cv2.contourArea)
    pixel_area = cv2.contourArea(lake_contour)
    pixel_to_meter = 0.5  
    real_area = pixel_area * (pixel_to_meter ** 2)
    average_depth = 5  
    lake_volume = real_area * average_depth  

    cv2.drawContours(image_cv, [lake_contour], -1, (0, 255, 0), 2)

    return image_cv, real_area, lake_volume

st.title("Lake Area & Capacity Estimator")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image, real_area, lake_volume = process_image(image)

    if processed_image is not None:
        st.image(processed_image, caption="Detected Lake", use_column_width=True)
        st.success(f"**Lake Area:** {real_area:.2f} m²\n\n**Lake Capacity:** {lake_volume:.2f} m³")
