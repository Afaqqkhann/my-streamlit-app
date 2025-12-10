import streamlit as st
from ultralytics import YOLO
import cv2
import os
import uuid
import tempfile
import traceback

# ---------- LOAD MODEL ----------
model = YOLO("yolov8n.pt")

st.set_page_config(page_title="Vehicle Detection", layout="centered")
st.title("🚗 YOLOv8 Vehicle Detection")

# ---------- FOLDERS ----------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    if st.button("Process Video"):

        try:
            # Save uploaded file
            input_name = str(uuid.uuid4()) + ".mp4"
            input_path = os.path.join(UPLOAD_FOLDER, input_name)

            with open(input_path, "wb") as f:
                f.write(uploaded_file.read())

            # Output file
            output_name = "output_" + input_name
            output_path = os.path.join(OUTPUT_FOLDER, output_name)

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error("Cannot read video file")
                st.stop()

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_limit = 500

            count = 0
            progress = st.progress(0)

            while True:
                ret, frame = cap.read()
                if not ret or count > frame_limit:
                    break

                result = model(frame, verbose=False)
                frame = result[0].plot()
                out.write(frame)

                count += 1
                progress.progress(min(count / frame_limit, 1.0))

            cap.release()
            out.release()

            st.success("✅ Processing complete!")

            # Display Video
            st.video(output_path)

            # Download Button
            with open(output_path, "rb") as f:
                st.download_button(
                    label="📥 Download Video",
                    data=f,
                    file_name=output_name,
                    mime="video/mp4"
                )

        except Exception as e:
            st.error("❌ Error occurred")
            st.text(traceback.format_exc())
