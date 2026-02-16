from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from google.colab.patches import cv2_imshow

def detect_mask(image_path, 
                face_detector_path="/content/Face-Mask-Detection/face_detector",
                model_path="/content/Face-Mask-Detection/mask_detector_new.h5",
                confidence_threshold=0.5):

    # Load face detector model
    print("[INFO] Loading face detector model...")
    prototxtPath = os.path.join(face_detector_path, "deploy.prototxt")
    weightsPath = os.path.join(face_detector_path, "res10_300x300_ssd_iter_140000.caffemodel")

    if not (os.path.exists(prototxtPath) and os.path.exists(weightsPath)):
        raise FileNotFoundError("❌ Face detector files missing. Check path.")

    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # Load mask classifier
    print("[INFO] Loading mask detector model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Mask detector model (.h5) missing. Check path.")

    model = load_model(model_path)

    # Load input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("❌ Image not found. Check image path.")

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX, startY = max(0,startX), max(0,startY)
            endX, endY = min(w-1,endX), min(h-1,endY)

            face = image[startY:endY, startX:endX]

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face)[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0,255,0) if label == "Mask" else (0,0,255)
            score = max(mask, withoutMask) * 100
            label = f"{label}: {score:.2f}%"

            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    # ✅ Show result in Colab
    cv2_imshow(image)


# ---- RUN IT ----
detect_mask("/content/Face-Mask-Detection/images/pic1.jpeg")


