import cv2
import numpy as np

# ── Config ─────────────────────────────────────────
output_path = 'test_video.mp4'
width, height = 640, 480
fps = 30
duration = 60  # seconds
total_frames = fps * duration

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("🎥 Creating test video...")

for i in range(total_frames):
    frame = np.ones((height, width, 3), dtype=np.uint8) * 50

    # ── Draw Face ───────────────────────────────────
    # Head
    cv2.ellipse(frame, (320, 240), (120, 150), 0, 0, 360, (210, 180, 140), -1)

    # Eyebrows
    cv2.line(frame, (240, 170), (290, 165), (80, 50, 30), 4)
    cv2.line(frame, (350, 165), (400, 170), (80, 50, 30), 4)

    # Nose
    cv2.ellipse(frame, (320, 250), (15, 10), 0, 0, 360, (180, 140, 110), -1)

    # Mouth
    cv2.ellipse(frame, (320, 300), (40, 15), 0, 0, 180, (150, 80, 80), 2)

    # ── Eye Animation ───────────────────────────────
    # Pattern: open 2sec → closing 1sec → closed 2sec → open 1sec
    cycle = i % (fps * 6)  # 6 second cycle

    if cycle < fps * 2:
        # Eyes OPEN
        eye_open = 1.0
    elif cycle < fps * 3:
        # Eyes CLOSING gradually
        eye_open = 1.0 - ((cycle - fps * 2) / fps)
    elif cycle < fps * 5:
        # Eyes CLOSED
        eye_open = 0.0
    else:
        # Eyes OPENING gradually
        eye_open = (cycle - fps * 5) / fps

    eye_h = max(2, int(18 * eye_open))

    # Left Eye White
    cv2.ellipse(frame, (265, 210), (28, 18), 0, 0, 360, (255, 255, 255), -1)
    # Right Eye White
    cv2.ellipse(frame, (375, 210), (28, 18), 0, 0, 360, (255, 255, 255), -1)

    # Left Iris
    cv2.ellipse(frame, (265, 210), (14, min(14, eye_h+4)), 0, 0, 360, (80, 60, 40), -1)
    # Right Iris
    cv2.ellipse(frame, (375, 210), (14, min(14, eye_h+4)), 0, 0, 360, (80, 60, 40), -1)

    # Left Pupil
    cv2.ellipse(frame, (265, 210), (6, min(6, eye_h)), 0, 0, 360, (10, 10, 10), -1)
    # Right Pupil
    cv2.ellipse(frame, (375, 210), (6, min(6, eye_h)), 0, 0, 360, (10, 10, 10), -1)

    # Eyelids closing
    if eye_open < 0.5:
        lid_h = int(18 * (1 - eye_open))
        # Left eyelid
        cv2.ellipse(frame, (265, 210), (28, lid_h), 0, 0, 360, (210, 180, 140), -1)
        # Right eyelid
        cv2.ellipse(frame, (375, 210), (28, lid_h), 0, 0, 360, (210, 180, 140), -1)

    # ── Status Text ─────────────────────────────────
    if eye_open > 0.5:
        status = "AWAKE"
        color  = (0, 255, 0)
    else:
        status = "DROWSY"
        color  = (0, 0, 255)

    cv2.putText(frame, f"Status: {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Frame: {i}/{total_frames}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    out.write(frame)

    if i % (fps * 5) == 0:
        print(f"  Progress: {i//fps}/{duration} seconds done...")

out.release()
print(f"\n✅ Test video created: {output_path}")
print(f"   Duration : {duration} seconds")
print(f"   Size     : {width}x{height}")
print(f"   FPS      : {fps}")