import cv2
import numpy as np
from tensorflow.keras.models import load_model
import winsound

# ── Config ─────────────────────────────────────────
MODEL_PATH   = 'model/drowsiness_model.h5'
ALERT_FRAMES = 15
OPEN_SCORE   = 0.5

print("Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded!")

# ── Only face cascade needed! ───────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

if not cap.isOpened():
    print("❌ Cannot open webcam!")
    exit()

score        = 0
frame_count  = 0
alert_count  = 0
paused       = False
pred_history = []

print("▶️  Running... Q = quit | SPACE = pause")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (800, 500))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(120, 120)
        )

        eye_status = "NO FACE"
        color      = (200, 200, 200)

        for (fx, fy, fw, fh) in faces[:1]:
            cv2.rectangle(frame, (fx,fy), (fx+fw,fy+fh), (0,255,255), 2)

            # ── Extract eye regions directly from face ──
            # Left eye region — top left quarter of face
            le_x1 = fx
            le_x2 = fx + fw//2
            le_y1 = fy + int(fh * 0.20)
            le_y2 = fy + int(fh * 0.50)

            # Right eye region — top right quarter of face
            re_x1 = fx + fw//2
            re_x2 = fx + fw
            re_y1 = fy + int(fh * 0.20)
            re_y2 = fy + int(fh * 0.50)

            # Draw eye region boxes
            cv2.rectangle(frame, (le_x1,le_y1), (le_x2,le_y2), (255,255,0), 1)
            cv2.rectangle(frame, (re_x1,re_y1), (re_x2,re_y2), (255,255,0), 1)
            cv2.putText(frame, "L", (le_x1, le_y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, "R", (re_x1, re_y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

            predictions = []

            # ── Predict both eye regions ────────────────
            for (x1,y1,x2,y2), label in [
                ((le_x1,le_y1,le_x2,le_y2), "L"),
                ((re_x1,re_y1,re_x2,re_y2), "R")
            ]:
                eye_crop = gray[y1:y2, x1:x2]
                if eye_crop.size == 0:
                    continue

                eye_img = cv2.resize(eye_crop, (80, 80))
                eye_img = eye_img / 255.0
                eye_img = eye_img.reshape(1, 80, 80, 1)
                pred    = model.predict(eye_img, verbose=0)[0][0]
                predictions.append(pred)

                ec = (0,255,0) if pred > OPEN_SCORE else (0,0,255)
                status = "O" if pred > OPEN_SCORE else "C"
                cv2.putText(frame,
                            f"{label}:{status} {pred:.2f}",
                            (x1, y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ec, 2)

            # ── Smoothed prediction ──────────────────────
            if predictions:
                avg_pred = np.mean(predictions)
                pred_history.append(avg_pred)
                if len(pred_history) > 5:
                    pred_history.pop(0)
                smooth = np.mean(pred_history)

                cv2.putText(frame, f"Pred: {smooth:.2f}",
                            (630, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255,255,255), 2)

                if smooth < OPEN_SCORE:
                    score += 2
                    eye_status = "DROWSY"
                    color = (0, 0, 255)
                else:
                    score = max(0, score - 1)
                    eye_status = "AWAKE"
                    color = (0, 255, 0)
            else:
                score = max(0, score - 1)
                eye_status = "SEARCHING..."
                color = (200, 200, 0)

        # ── Alert ────────────────────────────────────
        if score >= ALERT_FRAMES:
            alert_count += 1
            cv2.rectangle(frame, (0,0), (800,500), (0,0,255), 12)
            cv2.putText(frame, "DROWSINESS ALERT!",
                        (130,250), cv2.FONT_HERSHEY_DUPLEX,
                        1.6, (0,0,255), 3)
            cv2.putText(frame, "PLEASE TAKE A BREAK!",
                        (160,300), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,0,255), 2)
            winsound.Beep(1000, 300)
            score = 0

        # ── HUD ──────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (340,155), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"Status : {eye_status}",
                    (10,30),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, f"Score  : {score}/{ALERT_FRAMES}",
                    (10,65),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2)
        cv2.putText(frame, f"Alerts : {alert_count}",
                    (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,165,255), 2)
        cv2.putText(frame, f"Frame  : {frame_count}",
                    (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

        # ── Drowsiness Bar ───────────────────────────
        bar_w = min(300, int((score/ALERT_FRAMES)*300))
        bar_c = (0,255,0) if score < ALERT_FRAMES//2 else \
                (0,165,255) if score < ALERT_FRAMES else (0,0,255)
        cv2.rectangle(frame, (10,465), (310,485), (50,50,50), -1)
        cv2.rectangle(frame, (10,465), (10+bar_w,485), bar_c, -1)
        cv2.putText(frame, "Drowsiness Level",
                    (10,460), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200,200,200), 1)

    cv2.imshow("Driver Drowsiness Detection - LIVE", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
        print("⏸ Paused" if paused else "▶️  Resumed")

cap.release()
cv2.destroyAllWindows()

print(f"\n📊 SESSION SUMMARY")
print(f"   Total Frames : {frame_count}")
print(f"   Total Alerts : {alert_count}")