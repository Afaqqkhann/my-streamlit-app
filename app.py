from flask import Flask, request, send_file, render_template_string
from ultralytics import YOLO
import cv2
import os
import uuid
import traceback

app = Flask(__name__)

model = YOLO('yolov8n.pt')

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Vehicle Detection</title>
    <style>
        body {background:#0f2027;color:white;font-family:Arial;text-align:center;padding:40px;}
        .box {background:rgba(255,255,255,0.1);padding:30px;border-radius:15px;width:60%;margin:auto;}
        input, button {padding:12px;margin:10px;width:70%;font-size:16px;}
        button {background:#00ffcc;border:none;font-weight:bold;cursor:pointer;}
        a {color:yellow;font-size:20px;text-decoration:none;}
    </style>
</head>
<body>
<div class="box">
<h1>🚗 YOLOv8 Vehicle Detection</h1>
<form method="POST" action="/process" enctype="multipart/form-data">
<input type="file" name="video" required><br>
<button type="submit">Process Video</button>
</form>
{% if error %}
<p style="color:red;">{{ error }}</p>
{% endif %}
{% if download_link %}
<h3>✅ Processing complete</h3>
<a href="{{ download_link }}">Download Video</a>
{% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)

@app.route("/process", methods=["POST"])
def process_video():
    try:
        file = request.files.get("video")
        if not file:
            return render_template_string(HTML_PAGE, error="No file uploaded")

        input_name = str(uuid.uuid4()) + ".mp4"
        input_path = os.path.join(UPLOAD_FOLDER, input_name)
        file.save(input_path)

        output_name = "output_" + input_name
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return render_template_string(HTML_PAGE, error="Cannot read video file")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_limit = 500   # ✅ safety limit to avoid crash

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret or count > frame_limit:
                break

            result = model(frame, verbose=False)
            frame = result[0].plot()
            out.write(frame)

            count += 1

        cap.release()
        out.release()

        return render_template_string(HTML_PAGE, download_link=f"/download/{output_name}")

    except Exception as e:
        print(traceback.format_exc())
        return render_template_string(HTML_PAGE, error=str(e))

@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
