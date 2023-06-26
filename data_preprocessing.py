import os
import pickle

import cv2
import mediapipe as mp

classifier_set = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
                  11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
                  21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}


def process_data(data_dir):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.1)

    data = []
    labels = []
    total_count = 0

    for directory in os.listdir(data_dir):
        path = os.path.join(data_dir, directory)
        if not os.path.isdir(path):
            continue
        total_count += process_images(path, data, hands, total_count, directory, labels)
    print("Total Image Count: ", total_count)

    return data, labels


def process_images(path, data, hands, total_count, directory="", labels=None):
    if labels is None:
        labels = []
    count = 0
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        landmarks = hands.process(img_rgb)
        process_img(landmarks, data, labels, directory)
        count += 1
        print(count)
    return count


def process_img(landmarks, data, labels, directory):
    x_points = []
    y_points = []
    normalized_data = []
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


def save_data(data, labels):
    f = open('data.pickle', 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()
    print('Data saved')


def main():
    data, labels = process_data('./asl_alphabet_train')
    save_data(data, labels)


if __name__ == '__main__':
    main()
