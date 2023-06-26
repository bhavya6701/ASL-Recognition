import pickle

import cv2
import mediapipe as mp
import numpy as np
from keras import models

from data_preprocessing import classifier_set


def main():
    model = models.load_model('asl_recognition_model.h5')
    vid = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    while (True):
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        process_image_and_predict(cv2, frame, model, mp_hands, mp_drawing, mp_drawing_styles, hands)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

def process_image_and_predict(cv2, frame, model, mp_hands, mp_drawing, mp_drawing_styles, hands):
    data_aux, x_points, y_points = [], [], []

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
                data_aux.append(x - min(x_points))
                data_aux.append(y - min(y_points))

        x1 = int(min(x_points) * W) - 10
        y1 = int(min(y_points) * H) - 10

        x2 = int(max(x_points) * W) - 10
        y2 = int(max(y_points) * H) - 10

        data = np.asarray(data_aux)
        if data.shape[0] == 42:
            prediction = model.predict(data.reshape(1, 42))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            print(classifier_set[np.argmax(prediction)])
            cv2.putText(frame, classifier_set[np.argmax(prediction)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)


if __name__ == '__main__':
    main()