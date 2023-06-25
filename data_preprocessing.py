import os
import pickle

import cv2
import mediapipe as mp


def process_data():
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    # Load the data and split it into train and test sets
    DATA_DIR = './asl_alphabet_train'

    data = []
    labels = []

    for directory in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, directory)
        if not os.path.isdir(path):
            continue

        count = 0
        for img in os.listdir(path):
            normalized_data = []
            img_path = os.path.join(path, img)
            img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            landmarks = hands.process(img_rgb)

            x_points = []
            y_points = []

            if landmarks.multi_hand_landmarks:
                for hand in landmarks.multi_hand_landmarks:
                    for i in range(len(hand.landmark)):
                        x = hand.landmark[i].x
                        y = hand.landmark[i].y

                        x_points.append(x)
                        y_points.append(y)

                    for i in range(len(hand.landmark)):
                        x = hand.landmark[i].x
                        y = hand.landmark[i].y
                        normalized_data.append(x - min(x_points))
                        normalized_data.append(y - min(y_points))

                    data.append(normalized_data)
                    labels.append(directory)
                    normalized_data = []

            count += 1
            print(count)

    f = open('data.pickle', 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()
    print('Data saved')

def main():
    process_data()


if __name__ == '__main__':
    main()
