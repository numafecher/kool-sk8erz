import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# pairs of landmark ids to join with lines when drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


class FindHands():
    def __init__(self, detection_con=0.5, tracking_con=0.5):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=detection_con,
            min_tracking_confidence=tracking_con,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.ts = 0

    def getPosition(self, img, indexes, hand_no=0, draw=True):
        if isinstance(indexes, int):
            indexes = (indexes,)
        lst = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self.ts += 33  # timestamps must keep increasing in VIDEO mode
        results = self.landmarker.detect_for_video(mp_img, self.ts)

        if results.hand_landmarks and len(results.hand_landmarks) > hand_no:
            h, w, _ = img.shape
            landmarks = results.hand_landmarks[hand_no]
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            for id in indexes:
                lst.append(points[id])

            if draw:
                for a, b in HAND_CONNECTIONS:
                    cv2.line(img, points[a], points[b], (255, 255, 255), 2)
                for p in points:
                    cv2.circle(img, p, 4, (255, 0, 0), cv2.FILLED)
        return lst

    def _finger_up(self, img, ids, hand_no):
        pos = self.getPosition(img, ids, hand_no=hand_no, draw=False)
        if len(pos) < 2:
            return "NO HAND FOUND"
        return pos[0][1] >= pos[1][1]

    def index_finger_up(self, img, hand_no=0):
        return self._finger_up(img, (6, 8), hand_no)

    def middle_finger_up(self, img, hand_no=0):
        return self._finger_up(img, (10, 12), hand_no)

    def ring_finger_up(self, img, hand_no=0):
        return self._finger_up(img, (14, 16), hand_no)

    def little_finger_up(self, img, hand_no=0):
        return self._finger_up(img, (18, 20), hand_no)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    hands = FindHands()
    while True:
        succeed, img = cap.read()
        if not succeed:
            continue
        lst = hands.getPosition(img, (8,))
        for pt in lst:
            cv2.circle(img, pt, 8, (0, 255, 0), cv2.FILLED)
        cv2.imshow("Image", img)
        if cv2.waitKey(10) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()