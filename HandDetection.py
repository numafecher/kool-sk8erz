import cv2
import HandTrackingModule as htm

wCam, hCam = 1000, 1000
cap = cv2.VideoCapture(0)

detector = htm.FindHands(detection_con=0.79)

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index finger
    (0, 9), (9,10), (10,11), (11,12),       # Middle finger
    (0,13), (13,14), (14,15), (15,16),      # Ring finger
    (0,17), (17,18), (18,19), (19,20)       # Little finger
]

while True:
    success, img = cap.read()
    if not success:
        break

    lmlist = detector.getPosition(img, list(range(21)), draw=False)

    if len(lmlist) != 0:
        finger = []

        if lmlist[5][0] > lmlist[17][0]:
            handType = "Right"
        else:
            handType = "Left"



        if handType == "Right":
            finger.append(1 if lmlist[4][0] > lmlist[3][0] else 0)

        else:
            finger.append(1 if lmlist[4][0] < lmlist[3][0] else 0)


        finger.append(detector.index_finger_up(img))
        finger.append(detector.middle_finger_up(img))
        finger.append(detector.ring_finger_up(img))
        finger.append(detector.little_finger_up(img))

        TotalFingers = finger.count(1)
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(TotalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)

        for lm in lmlist:
            x, y = lm[0], lm[1]
            cv2.circle(img, (x, y), 5, (0, 0, 255), cv2.FILLED)

        for connection in connections:
            x1, y1 = lmlist[connection[0]][0], lmlist[connection[0]][1]
            x2, y2 = lmlist[connection[1]][0], lmlist[connection[1]][1]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


    cv2.imshow("image", img)
    cv2.waitKey(1)
cv2.release()
cv2.destroyAllWindows()