# ============================================================
# REAL-TIME FACE MASK + HAND/FINGER DETECTION (Optimized)
# ============================================================

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import cv2
import imutils
import os
from collections import deque
import mediapipe as mp

# ---- Paths ----
face_model_path = "D:/work/vidut_vega/mask_model/face_detector"
mask_model_path = "D:/work/vidut_vega/mask_model/mask_detector_quant.tflite"

# ---- Check if face model files exist ----
prototxtPath = os.path.join(face_model_path, "deploy.prototxt")
weightsPath = os.path.join(face_model_path, "res10_300x300_ssd_iter_140000.caffemodel")
if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath):
    raise FileNotFoundError("Face detector model files missing!")

# ---- Load Face Detector ----
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# ---- Load TFLite Mask Detector ----
if not os.path.exists(mask_model_path):
    raise FileNotFoundError("Mask TFLite model not found!")
interpreter = tf.lite.Interpreter(model_path=mask_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---- MediaPipe Hands ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---- Prediction with TFLite ----
def predict_tflite(interpreter, faces):
    preds = []
    for face in faces:
        input_data = np.expand_dims(face, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        preds.append(output[0])
    return np.array(preds)

# ---- Detection function ----
def detect_and_predict_mask(frame, faceNet, interpreter, conf_thresh=0.5, min_face=40):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, preds = [], [], []

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < conf_thresh:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Skip tiny faces
        if (endX - startX < min_face) or (endY - startY < min_face):
            continue

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

    if faces:
        faces = np.array(faces, dtype="float32")
        preds = predict_tflite(interpreter, faces)

    return locs, preds

# ---- Open Webcam ----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

print("🎥 Webcam feed started. Press 'q' to quit.")

# ---- Parameters ----
padding = 20
confidence_threshold = 0.7
temporal_window = 5  # frames to smooth predictions
pred_history = deque(maxlen=temporal_window)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)

    # --- Face mask detection ---
    locs, preds = detect_and_predict_mask(frame, faceNet, interpreter)

    # --- Hand detection ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)

    hand_boxes = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
            hand_boxes.append((x_min, y_min, x_max, y_max))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

    frame_preds = []

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # ---- Apply padding ----
        startX = max(0, startX - padding)
        startY = max(0, startY - padding)
        endX = min(frame.shape[1] - 1, endX + padding)
        endY = min(frame.shape[0] - 1, endY + padding)

        # ---- Check for hand occlusion ----
        occluded = False
        for (hx_min, hy_min, hx_max, hy_max) in hand_boxes:
            if (hx_min < endX and hx_max > startX and hy_min < endY and hy_max > startY):
                occluded = True
                break

        # ---- Threshold logic ----
        if mask > withoutMask and mask > confidence_threshold:
            label = "Mask"
            color = (0, 255, 0)
        elif withoutMask > confidence_threshold:
            label = "No Mask"
            color = (0, 0, 255)
        else:
            label = "Uncertain"
            color = (0, 255, 255)

        if occluded:
            label += " (Hand Occluded)"
            color = (0, 165, 255)  # orange for occlusion

        frame_preds.append((startX, startY, endX, endY, label, max(mask, withoutMask), color))

    # ---- Temporal smoothing to reduce flicker ----
    pred_history.append(frame_preds)
    avg_preds = {}
    for f_preds in pred_history:
        for (sx, sy, ex, ey, lbl, conf, clr) in f_preds:
            key = (sx, sy, ex, ey)
            if key not in avg_preds:
                avg_preds[key] = []
            avg_preds[key].append((lbl, conf, clr))

    for key, vals in avg_preds.items():
        sx, sy, ex, ey = key
        lbl = max(vals, key=lambda x: x[1])[0]
        clr = max(vals, key=lambda x: x[1])[2]
        conf = max(vals, key=lambda x: x[1])[1]
        label_text = f"{lbl}: {conf*100:.2f}%"
        cv2.putText(frame, label_text, (sx, sy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2)
        cv2.rectangle(frame, (sx, sy), (ex, ey), clr, 2)

    cv2.imshow("Face Mask + Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam feed stopped.")
