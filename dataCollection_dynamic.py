import cv2
import numpy as np
import os
import mediapipe as mp
import sys
# --- 1. Configuration ---
# Directory where the data will be saved (create if it doesn't exist)
DATA_PATH = os.path.join('MP_Data_Dynamic')

# Actions (signs) to collect. Start with J and Z.
ACTIONS = np.array(['J', 'Z', 'HELLO', 'I_LOVE_YOU'])

# Parameters for sequence collection
SEQUENCE_LENGTH = 30  # Number of frames to capture for one instance of a sign
NUM_SEQUENCES = 30  # Total number of sign instances (sequences) to collect per action

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    """Processes the image and returns the MediaPipe results."""
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    # Convert back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    """Draws hand keypoints onto the image."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(250, 44, 110), thickness=2, circle_radius=2)
                                      )


def extract_keypoints(results):
    """
    Extracts 3D (x, y, z) coordinates for all 21 hand landmarks.
    Returns a flattened array of 63 features (21 * 3).
    """
    if results.multi_hand_landmarks:
        # Assuming only one hand is visible for simplicity
        hand = results.multi_hand_landmarks[0].landmark

        # Flatten the coordinates into a single vector
        hand_keypoints = np.array([[res.x, res.y, res.z] for res in hand]).flatten()

        return hand_keypoints
    else:
        # Return an array of zeros if no hand is detected
        return np.zeros(21 * 3)

    # --- Directory Setup ---


for action in ACTIONS:
    for sequence in range(NUM_SEQUENCES):
        # Create directories for each sequence (e.g., MP_Data_Dynamic/J/0, MP_Data_Dynamic/J/1, ...)
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# --- Main Data Collection Loop ---
# ... (Configuration, MediaPipe Setup, Helper Functions defined above) ...

# --- Main Data Collection Loop ---
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hand_detector:
    for action in ACTIONS:
        for sequence_idx in range(NUM_SEQUENCES):

            sequence_frames = []  # Start a new list for each sequence

            # --- 1. Collection Start/Wait Period (WAIT FOR KEY PRESS) ---

            # Use an indefinite loop that breaks only on key press
            while True:
                ret, frame = cap.read()

                # Check for successful frame read
                if not ret:
                    continue

                # Process the frame to display landmarks while waiting
                current_image, current_results = mediapipe_detection(frame, hand_detector)
                draw_styled_landmarks(current_image, current_results)

                # Display instructions to the user
                cv2.putText(current_image, 'READY? PRESS "S" TO START RECORDING!', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(current_image, f'Action: {action}', (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(current_image, f'Sequence: {sequence_idx + 1}/{NUM_SEQUENCES}', (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', current_image)

                # Check for 's' key press (Start recording)
                key = cv2.waitKey(10) & 0xFF
                if key == ord('s'):
                    break
                # Allow user to quit early
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)  # Exit the function/script

            # --- 2. Sequence Recording Loop (30 Frames) ---
            for frame_idx in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()

                if not ret:
                    continue

                current_image, current_results = mediapipe_detection(frame, hand_detector)
                draw_styled_landmarks(current_image, current_results)

                # Extract and Store Keypoints
                keypoints = extract_keypoints(current_results)
                sequence_frames.append(keypoints)

                # Display progress and status
                cv2.putText(current_image, f'RECORDING: {action}', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(current_image, f'FRAME {frame_idx + 1}/{SEQUENCE_LENGTH}', (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', current_image)

                # We still need a waitKey here for the video feed to update
                cv2.waitKey(1)

                # --- 3. Save the entire sequence ---
            save_path = os.path.join(DATA_PATH, action, str(sequence_idx), f'{action}_{sequence_idx}.npy')
            np.save(save_path, np.array(sequence_frames))
            print(f'✅ Saved Sequence: {save_path}')

            # Small break after saving
            cv2.waitKey(500)

    cap.release()
    cv2.destroyAllWindows()