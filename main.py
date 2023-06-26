import pickle

import cv2
import mediapipe as mp
import numpy as np
from keras import models

from data_preprocessing import classifier_set


def main():
    model = models.load_model('asl_recognition_model.h5')
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    while (True):
        ret, frame = vid.read()
        process_image_and_predict(cv2, frame, model, mp_hands, mp_drawing, mp_drawing_styles, hands)
        cv2.imshow('ASL Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

def process_image_and_predict(cv2, frame, model, mp_hands, mp_drawing, mp_drawing_styles, hands):
    normalized_data, x_points, y_points = [], [], []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_points.append(x)
                y_points.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                normalized_data.append(x - min(x_points))
                normalized_data.append(y - min(y_points))

        x1 = int(min(x_points) * W) - 10
        y1 = int(min(y_points) * H) - 10

        x2 = int(max(x_points) * W) - 10
        y2 = int(max(y_points) * H) - 10

        data = np.asarray(normalized_data)
        if data.shape[0] == 42:
            prediction = model.predict(data.reshape(1, 42))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, classifier_set[np.argmax(prediction)], (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)


if __name__ == '__main__':
    main()