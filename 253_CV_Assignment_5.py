import numpy as np
import cv2
import streamlit as st
from PIL import Image

def main_loop():
    st.title("CV Assignment 5")
    st.subheader("Implement image segmentation using grabcut algorithm for multiple objects in image")
    st.text("Name: Kshitija Lade")
    st.text("Roll No.: 253")
    st.text("PRN: 0120190090")
    st.text("Batch: CV2")

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    mask = np.zeros(original_image.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (100, 30, 421, 378)
    cv2.grabCut(original_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = original_image * mask2[:, :, np.newaxis]

    st.text("Grabcut Algorithm")
    st.text("Original Image")
    st.image(original_image)
    st.text("Grabcut Image")
    st.image(img)

if __name__ == '__main__':
    main_loop()