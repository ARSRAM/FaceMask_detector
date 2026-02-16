# ============================================================
# FACE MASK DETECTION — COLAB-FRIENDLY MP4 OUTPUT
# ============================================================

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import imutils
import os
from IPython.display import HTML, display
from base64 import b64encode

# ---- Paths ----
face_model_path = "/content/Face-Mask-Detection/face_detector"
mask_model_path = "/content/Face-Mask-Detection/mask_detector_new.h5"
video_path = "/content/drive/MyDrive/Hackathon_DSAI/viduyt_vega/images/4.mp4"
output_path = "/content/drive/MyDrive/Hackathon_DSAI/viduyt_vega/images/output_video/mask_detected_video2.mp4"

# ---- Load models ----
prototxtPath = os.path.join(face_model_path, "deploy.prototxt")
weightsPath = os.path.join(face_model_path, "res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(mask_model_path)

# ---- Open video ----
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Video file not found or cannot be opened.")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ---- Determine output dimensions (resize width) ----
resize_width = 600
resize_height = int((resize_width / width) * height)

# ---- MP4 Codec (use H.264 for better compatibility) ----
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
out = cv2.VideoWriter(output_path, fourcc, fps, (resize_width, resize_height))

# Check if VideoWriter opened successfully
if not out.isOpened():
    print("⚠️ Trying alternative codec...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (resize_width, resize_height))

# ---- Detection function ----
def detect_and_predict_mask(frame, faceNet, maskNet, confidence=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, preds = [], [], []

    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w-1, endX), min(h-1, endY)

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return locs, preds

# ---- Process video ----
frame_count = 0
print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame consistently
    frame = imutils.resize(frame, width=resize_width)
    
    locs, preds = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
        cv2.putText(frame, label_text, (startX, startY-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    out.write(frame)
    frame_count += 1
    
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...", end='\r')

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n✅ Processed {frame_count} frames. Video saved to {output_path}")

# ---- Verify file exists and has content ----
if os.path.exists(output_path):
    file_size = os.path.getsize(output_path)
    print(f"📁 Output file size: {file_size / (1024*1024):.2f} MB")
else:
    print("❌ Output file was not created!")

# ---- Display video in Colab ----
try:
    with open(output_path, 'rb') as f:
        mp4 = f.read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display(HTML(f"""
    <video width="800" controls>
          <source src="{data_url}" type="video/mp4">
          Your browser does not support the video tag.
    </video>
    """))
    print("🎥 Video player displayed above")
except Exception as e:
    print(f"❌ Error displaying video: {e}")