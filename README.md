# 🧠 Real-Time Face Mask Detection with Hand Occlusion Awareness

### Deep Learning | Computer Vision | Transfer Learning | TensorFlow | OpenCV | MediaPipe

A robust real-time Face Mask Detection system built using **MobileNetV2 Transfer Learning**, capable of detecting mask compliance and identifying **hand occlusion over the face** to improve reliability in real-world environments.

🎯 **Validation Accuracy: 98%**

---

## 📌 Project Highlights

- 🚀 Real-time face mask detection via webcam  
- 🧠 Transfer Learning using MobileNetV2 (ImageNet pretrained)  
- ✋ Hand and finger detection using MediaPipe  
- 🛡️ Hand occlusion awareness  
- 📊 Achieved 98% accuracy  
- ⚡ Optimized for real-time inference  
- 📈 Training visualization available  

---

## 🧠 Model Architecture

This project uses **MobileNetV2 Transfer Learning**.

### Pipeline

```
Input Image (224×224×3)
        ↓
MobileNetV2 (Pretrained on ImageNet)
        ↓
AveragePooling Layer
        ↓
Flatten Layer
        ↓
Dense Layer (128, ReLU)
        ↓
Dropout Layer (0.5)
        ↓
Softmax Output Layer
        ↓
Prediction: Mask / No Mask
```

---

## 📊 Dataset Information

### Dataset Summary

| Class         | Number of Images |
|---------------|------------------|
| With Mask     | 2162             |
| Without Mask  | 1930             |
| **Total**     | **4092**         |

### Dataset Characteristics

- Labeled dataset  
- Balanced class distribution  
- Real-world images  
- Suitable for supervised learning  

---

## 📈 Model Performance

```
precision    recall  f1-score   support

with_mask       0.96      0.99      0.98       433
without_mask    0.99      0.96      0.97       386

accuracy                           0.98       819
macro avg       0.98      0.97      0.98       819
weighted avg    0.98      0.98      0.98       819
```

🎯 **Final Accuracy: 98%**

---

## ⚙️ Technologies Used

- Python  
- TensorFlow / Keras  
- MobileNetV2  
- OpenCV  
- MediaPipe  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## 🎥 Real-Time Detection

Run:

```
python test5.py
```

This will:

- Start webcam  
- Detect faces  
- Predict mask status  
- Detect hand occlusion  
- Display real-time output  

---

## 🔍 How It Works

1. Webcam captures frame  
2. OpenCV detects faces  
3. Face is preprocessed  
4. MobileNetV2 predicts mask status  
5. MediaPipe detects hands  
6. Occlusion logic applied  
7. Results displayed  

---

## 🎯 Applications

- Smart surveillance systems  
- Public safety monitoring  
- Healthcare compliance  
- Airport security  
- Workplace safety  

---

## 🧠 ML Concepts Used

- Transfer Learning  
- CNN (Convolutional Neural Networks)  
- Image Classification  
- Binary Classification  
- Data Augmentation  
- Real-time inference  

---

## 🔮 Future Improvements

- Web deployment using Flask  
- Edge deployment  
- Face recognition integration  
- Model optimization  

---


