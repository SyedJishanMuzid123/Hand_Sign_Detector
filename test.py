import cv2
import numpy as np
import math
import os
import tensorflow as tf
import mediapipe as mp
# Import CVZone modules for HandDetector and Classifier (Static Model)
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# --- 1. CONFIGURATION & MODEL LOADING ---
# Dynamic Model Config (from Code 1)
SEQUENCE_LENGTH = 30  # Must match training sequence length
# Static Model Config (from Code 3)
offset = 20
imgSize = 300

# Load Models
# The CVZone Classifier handles loading the Keras model for STATIC signs
classifier_static = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels_static = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]  # Your static signs

# Load Dynamic LSTM Model and Labels
try:
    model_dynamic = tf.keras.models.load_model('Model/model_dynamic_lstm.h5')
    # Load your dynamic labels (e.g., J, Z, HELLO, I_LOVE_YOU)
    labels_dynamic = np.load('Model/labels_dynamic.npy')
except:
    print("Warning: Could not load dynamic model. Only static prediction will run.")
    model_dynamic = None

# --- 2. MEDIAPIPE & HELPER FUNCTIONS (From Code 1) ---
# Required for Dynamic Keypoint Extraction
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    """Processes the image and returns the MediaPipe results."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    """Extracts 63 keypoints (21 * 3) from the first detected hand."""
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0].landmark
        hand_keypoints = np.array([[res.x, res.y, res.z] for res in hand]).flatten()
        return hand_keypoints
    else:
        return np.zeros(21 * 3)


def draw_styled_landmarks(image, results):
    """Draws hand keypoints onto the image (for visualization/debugging)."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


# --- 3. MAIN REAL-TIME INFERENCE LOOP ---
cap = cv2.VideoCapture(0)
detector_cvzone = HandDetector(maxHands=1)  # Used for CVZone cropping
sequence_buffer = []  # Buffer to store 30 frames of keypoints for dynamic model

# Use MediaPipe Hand Detector for keypoint extraction
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hand_detector_mp:
    while True:
        success, img = cap.read()
        if not success: continue

        # 1. Image Processing for STATIC Model (CVZone Bounding Box)
        imgOutput = img.copy()
        hands_cvzone, img_cvzone = detector_cvzone.findHands(img, draw=False)  # Get bounding box

        # 2. Keypoint Processing for DYNAMIC Model (MediaPipe)
        img_mp, results_mp = mediapipe_detection(img, hand_detector_mp)
        draw_styled_landmarks(imgOutput, results_mp)  # Draw MP landmarks on output frame
        keypoints = extract_keypoints(results_mp)

        # Update Dynamic Sequence Buffer
        sequence_buffer.append(keypoints)
        sequence_buffer = sequence_buffer[-SEQUENCE_LENGTH:]  # Keep only the last 30 frames

        # Initialize prediction variables
        predicted_sign = "WAITING..."

        # --- PREDICTION LOGIC ---

        # A. DYNAMIC (LSTM) PREDICTION (If buffer is full and model loaded)
        if model_dynamic is not None and len(sequence_buffer) == SEQUENCE_LENGTH and np.sum(sequence_buffer[-1]) != 0:
            input_data = np.expand_dims(sequence_buffer, axis=0)  # Reshape to (1, 30, 63)
            res = model_dynamic.predict(input_data, verbose=0)[0]
            dynamic_index = np.argmax(res)
            dynamic_confidence = res[dynamic_index]

            # Check if dynamic prediction is strong (e.g., > 80% confidence)
            if dynamic_confidence > 0.8:
                predicted_sign = labels_dynamic[dynamic_index]

        # B. STATIC (CNN) PREDICTION (Fallback or primary for static signs)
        # Only run static prediction if a hand is detected by CVZone and dynamic was not strong
        if 'predicted_sign' not in locals() or predicted_sign == "WAITING...":
            if hands_cvzone:
                hand = hands_cvzone[0]
                x, y, w, h = hand['bbox']

                # Hand Standardization (Same logic as Code 2 and 3)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[max(0, y - offset):y + h + offset, max(0, x - offset):x + w + offset]

                if imgCrop.size != 0:
                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    # Run Static Prediction
                    _, static_index = classifier_static.getPrediction(imgWhite, draw=False)
                    predicted_sign = labels_static[static_index]

        # --- DISPLAY RESULTS ---

        # Display Prediction on Image
        cv2.putText(imgOutput, predicted_sign, (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 255, 0), 2)

        # Optionally display the static cropped image for debugging
        # if 'imgWhite' in locals():
        #     cv2.imshow('ImageWhite', imgWhite)

        cv2.imshow('Image', imgOutput)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()