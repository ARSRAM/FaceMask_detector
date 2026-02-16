# ============================================================
# REAL-TIME FACE MASK + HAND/FINGER DETECTION (.H5 MODEL)
# ============================================================

import os
import cv2
import imutils
import numpy as np
import tensorflow as tf
from collections import deque
import mediapipe as mp

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ------------------------------------------------------------
# Disable oneDNN warning (optional)
# ------------------------------------------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
face_model_path = "D:/work/vidut_vega/mask_model/face_detector"
mask_model_path = "D:/work/vidut_vega/mask_model/mask_detector_new.h5"

# ------------------------------------------------------------
# CHECK FACE DETECTOR FILES
# ------------------------------------------------------------
prototxtPath = os.path.join(face_model_path, "deploy.prototxt")
weightsPath = os.path.join(face_model_path, "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath):
    raise FileNotFoundError("❌ Face detector files missing")

# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------
print("🔹 Loading face detector...")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("🔹 Loading mask detector (.h5)...")
mask_model = tf.keras.models.load_model(mask_model_path, compile=False)

# ------------------------------------------------------------
# MEDIAPIPE HANDS
# ------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------------------------------------
# FACE + MASK DETECTION FUNCTION
# ------------------------------------------------------------
def detect_and_predict_mask(frame, faceNet, mask_model,
                            conf_thresh=0.3, min_face=40):

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < conf_thresh:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        if (endX - startX < min_face) or (endY - startY < min_face):
            continue

        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(w - 1, endX), min(h - 1, endY)

        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((startX, startY, endX, endY))

    preds = []
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_model.predict(faces, batch_size=16, verbose=0)

    return locs, preds

# ------------------------------------------------------------
# WEBCAM
# ------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam")

print("🎥 Webcam started. Press 'q' to quit.")

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
padding = 20
MIN_CONF = 0.35
temporal_window = 5
pred_history = deque(maxlen=temporal_window)

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)

    # ---- Face + mask detection ----
    locs, preds = detect_and_predict_mask(frame, faceNet, mask_model)

    # ---- Hand detection ----
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)

    hand_boxes = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            h, w, _ = frame.shape
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(xs) * w), int(max(xs) * w)
            y_min, y_max = int(min(ys) * h), int(max(ys) * h)
            hand_boxes.append((x_min, y_min, x_max, y_max))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                          (255, 255, 0), 2)

    frame_preds = []

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        startX = max(0, startX - padding)
        startY = max(0, startY - padding)
        endX = min(frame.shape[1] - 1, endX + padding)
        endY = min(frame.shape[0] - 1, endY + padding)

        # ---- Hand occlusion check ----
        occluded = any(
            hx1 < endX and hx2 > startX and hy1 < endY and hy2 > startY
            for (hx1, hy1, hx2, hy2) in hand_boxes
        )

        # ---- Correct classification logic ----
        if mask > withoutMask:
            label = "Mask"
            color = (0, 255, 0)
            conf = mask
        else:
            label = "No Mask"
            color = (0, 0, 255)
            conf = withoutMask

        if conf < MIN_CONF:
            label = "Uncertain"
            color = (0, 255, 255)

        if occluded:
            label += " (Hand Occluded)"
            color = (0, 165, 255)

        frame_preds.append((startX, startY, endX, endY, label, conf, color))

    # ---- Temporal smoothing ----
    pred_history.append(frame_preds)
    avg_preds = {}

    for f_preds in pred_history:
        for (sx, sy, ex, ey, lbl, conf, clr) in f_preds:
            key = (sx, sy, ex, ey)
            avg_preds.setdefault(key, []).append((lbl, conf, clr))

    for (sx, sy, ex, ey), vals in avg_preds.items():
        lbl, conf, clr = max(vals, key=lambda x: x[1])
        cv2.putText(frame, f"{lbl}: {conf*100:.2f}%",
                    (sx, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2)
        cv2.rectangle(frame, (sx, sy), (ex, ey), clr, 2)

    cv2.imshow("Face Mask + Hand Detection (.h5)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam stopped cleanly.")
