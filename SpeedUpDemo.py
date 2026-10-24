import time
import cv2
from HandTrackingModule import FindHands
from Gestures import PushDetector

# --- game feel (change these freely) ---
BOOST = 1.5        # speed gained per push
MAX_SPEED = 10.0
FRICTION = 1.2     # speed lost per second when you stop pushing

cap = cv2.VideoCapture(0)
hands = FindHands()
push = PushDetector()

speed = 0.0
pushes = 0
ground_x = 0.0
last = time.time()

while True:
    ok, img = cap.read()
    if not ok:
        continue

    now = time.time()
    dt = now - last
    last = now

    # one hand detection per frame, reuse the points for every gesture
    pts = hands.getPosition(img, PushDetector.LANDMARKS)

    if push.update(pts, now):
        pushes += 1
        speed = min(MAX_SPEED, speed + BOOST)

    speed = max(0.0, speed - FRICTION * dt)
    ground_x = (ground_x + speed * 60 * dt) % 40

    # ---- HUD ----
    h, w, _ = img.shape

    # speed bar
    cv2.rectangle(img, (20, 20), (220, 45), (255, 255, 255), 2)
    cv2.rectangle(img, (20, 20), (20 + int(200 * speed / MAX_SPEED), 45), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f"speed {speed:.1f}   pushes {pushes}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # live readout for tuning thresholds
    if push.reach is None:
        txt = "reach: no hand"
    else:
        txt = f"reach {push.reach:.2f}  (push > {push.on_thresh}, reset < {push.off_thresh})"
    cv2.putText(img, txt, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # scrolling ground so you can *feel* the speed
    for x in range(-40, w, 40):
        gx = int(x - ground_x)
        cv2.line(img, (gx, h - 25), (gx + 20, h - 25), (255, 255, 255), 4)

    cv2.imshow("Image", img)
    if cv2.waitKey(10) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
