import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import winsound

# ── Config ─────────────────────────────────────────
MODEL_PATH   = 'model/drowsiness_model.h5'
ALERT_FRAMES = 10
CLOSED_SCORE = 0.5

# ── Load Model ──────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded!")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ── Open Video ──────────────────────────────────────
VIDEO_PATH = 'test_video.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Cannot open video!")
    exit()

score       = 0
frame_count = 0
alert_count = 0
paused      = False

print("▶️  Running... SPACE=pause | Q=quit")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video ended!")
            break

        frame_count += 1
        frame = cv2.resize(frame, (800, 500))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        eye_status = "NO FACE"
        color      = (200, 200, 200)

        for (fx, fy, fw, fh) in faces:
            roi_gray  = gray[fy:fy+fh, fx:fx+fw]
            roi_color = frame[fy:fy+fh, fx:fx+fw]
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 200, 0), 2)

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            predictions = []

            for (ex, ey, ew, eh) in eyes[:2]:
                eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
                eye_img = cv2.resize(eye_img, (80, 80))
                eye_img = eye_img / 255.0
                eye_img = eye_img.reshape(1, 80, 80, 1)

                pred = model.predict(eye_img, verbose=0)[0][0]
                predictions.append(pred)

                eye_color = (0, 255, 0) if pred > CLOSED_SCORE else (0, 0, 255)
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), eye_color, 2)
                label = f"Open:{pred:.2f}" if pred > CLOSED_SCORE else f"Closed:{pred:.2f}"
                cv2.putText(roi_color, label, (ex, ey-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, eye_color, 1)

            if predictions:
                avg_pred = np.mean(predictions)
                if avg_pred < CLOSED_SCORE:
                    score += 2
                    eye_status = "DROWSY"
                    color = (0, 0, 255)
                else:
                    score = max(0, score - 1)
                    eye_status = "AWAKE"
                    color = (0, 255, 0)

        # ── Alert ───────────────────────────────────
        if score >= ALERT_FRAMES:
            alert_count += 1
            cv2.rectangle(frame, (0, 0), (800, 500), (0, 0, 255), 10)
            cv2.putText(frame, "DROWSINESS ALERT!", (170, 250),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
            winsound.Beep(1000, 200)

        # ── HUD ─────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, f"Status : {eye_status}",          (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, f"Score  : {score}/{ALERT_FRAMES}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(frame, f"Alerts : {alert_count}",          (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        cv2.putText(frame, f"Frame  : {frame_count}",          (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
        print("⏸ Paused" if paused else "▶️  Resumed")

cap.release()
cv2.destroyAllWindows()

print(f"\n📊 SUMMARY")
print(f"   Total Frames  : {frame_count}")
print(f"   Total Alerts  : {alert_count}")