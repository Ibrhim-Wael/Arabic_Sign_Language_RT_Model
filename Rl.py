import cv2
import numpy as np
import tensorflow as tf
import json
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque
import urllib.request
import os

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH       = r"C:\Users\ibrah\Downloads\arabic_sign_model_v2.keras"
CLASS_NAMES_PATH = r"C:\Users\ibrah\Downloads\class_names.json"

# MediaPipe hand landmark model — auto-downloaded if not present
MP_HAND_MODEL = "hand_landmarker.task"
MP_HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

IMG_SIZE             = (64, 64)
CONFIDENCE_THRESHOLD = 0.70
SMOOTH_FRAMES        = 8
PADDING              = 30
# ─────────────────────────────────────────────


def download_hand_model():
    if not os.path.exists(MP_HAND_MODEL):
        print("Downloading MediaPipe hand model (~9MB) ...")
        urllib.request.urlretrieve(MP_HAND_MODEL_URL, MP_HAND_MODEL)
        print("Download complete.")


def preprocess(roi):
    img = cv2.resize(roi, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)   # (1,64,64,3) BGR — matches training


def smoothed_prediction(history):
    if not history:
        return None, 0.0
    avg = np.mean(history, axis=0)
    idx = int(np.argmax(avg))
    return idx, float(avg[idx])


def get_bbox(landmarks, W, H):
    xs = [lm.x * W for lm in landmarks]
    ys = [lm.y * H for lm in landmarks]
    x1 = max(0,  int(min(xs) - PADDING))
    y1 = max(0,  int(min(ys) - PADDING))
    x2 = min(W,  int(max(xs) + PADDING))
    y2 = min(H,  int(max(ys) + PADDING))
    return x1, y1, x2, y2


def draw_landmarks(frame, landmarks, W, H):
    """Draw hand skeleton manually (since mp.solutions.drawing_utils is gone)."""
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17)
    ]
    pts = [(int(lm.x * W), int(lm.y * H)) for lm in landmarks]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (0, 180, 80), 1)
    for pt in pts:
        cv2.circle(frame, pt, 3, (255, 255, 255), -1)


def main():
    download_hand_model()

    print("Loading sign language model ...")
    sign_model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    print(f"Model loaded  |  {len(class_names)} classes")

    # ── MediaPipe HandLandmarker (new Tasks API) ──────────────────────
    base_options = mp_python.BaseOptions(model_asset_path=MP_HAND_MODEL)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    pred_history = deque(maxlen=SMOOTH_FRAMES)
    print("Controls:  Q -> quit   |   C -> clear buffer\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            draw_landmarks(frame, landmarks, W, H)

            x1, y1, x2, y2 = get_bbox(landmarks, W, H)

            # Crop from BGR frame — matches training
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                probs = sign_model.predict(preprocess(roi), verbose=0)[0]
                pred_history.append(probs)

                idx, confidence = smoothed_prediction(pred_history)

                box_color = (0, 220, 100) if confidence >= CONFIDENCE_THRESHOLD else (60, 60, 210)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                if confidence >= CONFIDENCE_THRESHOLD:
                    label = f"{class_names[idx]}  {int(confidence*100)}%"
                else:
                    label = "..."
                cv2.putText(frame, label,
                            (x1, max(y1 - 12, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, box_color, 2)

                # Top-3 sidebar
                avg_probs = np.mean(pred_history, axis=0)
                top3 = np.argsort(avg_probs)[::-1][:3]
                overlay = frame.copy()
                cv2.rectangle(overlay, (W - 210, 0), (W, 115), (15, 15, 15), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, "Top 3:", (W - 200, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
                for rank, i in enumerate(top3):
                    color = (255, 255, 255) if rank == 0 else (110, 110, 110)
                    cv2.putText(frame,
                                f"  {rank+1}. {class_names[i]}  {int(avg_probs[i]*100)}%",
                                (W - 200, 45 + rank * 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1)
        else:
            pred_history.clear()
            cv2.putText(frame, "No hand detected",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 210), 2)

        # Bottom banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, H - 30), (W, H), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "Arabic Sign Language  |  Q = quit   C = clear",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        cv2.imshow("Arabic Sign Language - Real Time", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            pred_history.clear()

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()