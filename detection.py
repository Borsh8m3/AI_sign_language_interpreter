import cv2
import mediapipe as mp
import numpy as np
import csv
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = input("Podaj literę: ")

f = open("gestures.csv", "a", newline="")
writer = csv.writer(f)

count = 0
max_samples = 200 

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    print("Zacznij pokazywać literę:", label)
    time.sleep(2)

    while count < max_samples:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                writer.writerow([label] + landmarks)
                count += 1

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"{label} ({count}/{max_samples})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Zbieranie danych", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

f.close()
cap.release()
cv2.destroyAllWindows()
print(f"Zebrano {count} próbek dla litery {label}")
