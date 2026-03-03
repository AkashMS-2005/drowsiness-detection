import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model/drowsiness_model.h5')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)

print("📷 Opening webcam...")
print("Watch the RAW prediction values!")
print("Eyes OPEN  → value should be CLOSE TO 1.0")
print("Eyes CLOSED → value should be CLOSE TO 0.0")
print("Press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 500))
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80,80))

    for (fx, fy, fw, fh) in faces:
        roi_gray  = gray[fy:fy+fh//2, fx:fx+fw]
        roi_color = frame[fy:fy+fh//2, fx:fx+fw]

        cv2.rectangle(frame, (fx,fy), (fx+fw,fy+fh), (0,255,255), 2)

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 3, minSize=(20,20))

        for (ex, ey, ew, eh) in eyes[:2]:
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_img = cv2.resize(eye_img, (80, 80))
            eye_img = eye_img / 255.0
            eye_img = eye_img.reshape(1, 80, 80, 1)

            pred = model.predict(eye_img, verbose=0)[0][0]

            # Show RAW value
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (255,255,0), 2)
            cv2.putText(roi_color, f"RAW:{pred:.3f}", (ex, ey-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # Print to terminal
            print(f"RAW Prediction: {pred:.4f} | Eyes are: {'OPEN' if pred > 0.5 else 'CLOSED'}")

    cv2.imshow("Check Model", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()