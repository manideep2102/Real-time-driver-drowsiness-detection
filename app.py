import cv2
import numpy as np
import mediapipe as mp
import time
import os
from collections import deque
import pygame

os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.models import load_model
from scipy.spatial import distance as dist

# ================================
# LOAD MODEL
# ================================
model = load_model("driver_drowsiness_cnn.h5")

# ================================
# INIT PYGAME SOUND (CONTROLLED)
# ================================
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("freesound_community-emergency-alarm-with-reverb-29431.wav")
alarm_channel = None

def start_alarm():
    global alarm_channel
    if alarm_channel is None or not alarm_channel.get_busy():
        alarm_channel = alarm_sound.play(-1)  # loop continuously

def stop_alarm():
    global alarm_channel
    if alarm_channel is not None:
        alarm_channel.stop()

# ================================
# MEDIAPIPE SETUP
# ================================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ================================
# TEMPORAL SETTINGS
# ================================
EAR_THRESHOLD = 0.25
DROWSY_TIME = 2.0
CNN_CONF_THRESHOLD = 0.6

EAR_WINDOW_SIZE = 7
ear_history = deque(maxlen=EAR_WINDOW_SIZE)

PRED_WINDOW = 10
prediction_history = deque(maxlen=PRED_WINDOW)

eye_close_start = None
closed_duration = 0

# ================================
# YAWN VARIABLES
# ================================
YAWN_THRESH = 5
NO_YAWN_THRESH = 10
yawn_counter = 0
is_yawning = False
yawn_frame_count = 0
no_yawn_frame_count = 0

# ================================
# START WEBCAM
# ================================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "ALERT"

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        def get_point(idx):
            return int(landmarks.landmark[idx].x * w), int(landmarks.landmark[idx].y * h)

        left_eye = [get_point(i) for i in LEFT_EYE]
        right_eye = [get_point(i) for i in RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # EAR smoothing
        ear_history.append(ear)
        smoothed_ear = np.mean(ear_history)

        # CNN prediction
        face_img = cv2.resize(frame, (224, 224))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        preds = model.predict(face_img, verbose=0)
        label = np.argmax(preds)
        confidence = np.max(preds)

        # Majority voting
        prediction_history.append(label)
        if len(prediction_history) == PRED_WINDOW:
            final_label = max(set(prediction_history), key=prediction_history.count)
        else:
            final_label = label

        # ================================
        # DROWSINESS DETECTION
        # ================================
        if smoothed_ear < EAR_THRESHOLD:
            if eye_close_start is None:
                eye_close_start = time.time()
            closed_duration = time.time() - eye_close_start

            if closed_duration >= DROWSY_TIME:
                status = "DROWSY"
        else:
            eye_close_start = None
            closed_duration = 0

                # ================================
        # YAWNING DETECTION (IMPROVED)
        # ================================
        if final_label == 2 and confidence > CNN_CONF_THRESHOLD:
            yawn_frame_count += 1
            no_yawn_frame_count = 0
        else:
            no_yawn_frame_count += 1

        # Confirm yawning start
        if yawn_frame_count >= YAWN_THRESH:
            is_yawning = True

        # Confirm yawning ended
        if no_yawn_frame_count >= NO_YAWN_THRESH:
            is_yawning = False
            yawn_frame_count = 0

        # Set status
        if is_yawning:
            status = "YAWNING"

        # ================================
        # CONTROL ALARM BASED ON STATUS
        # ================================
        if status in ["DROWSY", "YAWNING"]:
            start_alarm()
        else:
            stop_alarm()

        # ================================
        # UI DISPLAY
        # ================================
        cv2.putText(frame, f"EAR: {smoothed_ear:.3f}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Eyes Closed: {closed_duration:.2f}s", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame, f"Yawns: {yawn_counter}", (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    color = (0, 255, 0) if status == "ALERT" else (0, 0, 255)
    cv2.putText(frame, f"Status: {status}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Real-Time Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()