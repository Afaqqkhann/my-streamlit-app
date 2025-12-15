import streamlit as st
from ultralytics import YOLO
import cv2
import os

# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(page_title="Vehicle Counting", layout="wide")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "best.pt"   # ⬅ model must be in repo

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()
class_names = model.names

# =====================================
# PROCESS VIDEO
# =====================================
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    line_y = int(h * 0.55)
    counts = {name: 0 for name in class_names.values()}
    counted_ids = set()

    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    frame_no = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            conf=0.25,
            tracker="bytetrack.yaml"
        )

        cv2.line(frame, (50, line_y), (w-50, line_y), (0,0,255), 3)

        if results[0].boxes.id is not None:
            for box, tid, cls in zip(
                results[0].boxes.xyxy.cpu().numpy(),
                results[0].boxes.id.cpu().numpy(),
                results[0].boxes.cls.cpu().numpy()
            ):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                cls = int(cls)

                if cls not in class_names:
                    continue

                name = class_names[cls]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                if abs(cy-line_y) < 4 and tid not in counted_ids:
                    counts[name] += 1
                    counted_ids.add(tid)

        y = 30
        for k,v in counts.items():
            cv2.putText(frame,f"{k}:{v}",(20,y),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
            y += 25

        out.write(frame)
        frame_no += 1
        progress.progress(frame_no/frame_total)

    cap.release()
    out.release()
    progress.empty()
    return counts

# =====================================
# UI
# =====================================
st.title("🚗 Vehicle Counting System")

video = st.file_uploader("Upload Traffic Video", type=["mp4","avi","mov"])

if video:
    in_path = os.path.join(UPLOAD_FOLDER, video.name)
    out_path = os.path.join(OUTPUT_FOLDER, "processed_" + video.name)

    with open(in_path, "wb") as f:
        f.write(video.read())

    if st.button("▶ Process Video"):
        counts = process_video(in_path, out_path)

        st.video(out_path)
        st.subheader("Vehicle Counts")
        for k,v in counts.items():
            st.write(f"**{k}**: {v}")

        with open(out_path,"rb") as f:
            st.download_button(
                "⬇ Download Result Video",
                f,
                "vehicle_counted.mp4",
                "video/mp4"
            )
