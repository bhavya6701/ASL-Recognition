import keras
import mediapipe as mp
import numpy as np
from keras import models

from data_preprocessing import process_images
from interface import classifier_set


def test_model(model: keras.Model):
    path = './asl_alphabet_test'
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    data = []
    process_images(hands, path, data)
    data = np.array(data)

    for i in range(len(data)):
        result = model.predict(data[i])
        print("Predicted Value:", str(classifier_set[np.argmax(result)]))
        print("Accuracy:", str(int(max(result) * 100)) + "%")

def main():
    model = models.load_model('asl_recognition_model.h5')
    test_model(model)


if __name__ == '__main__':
    main()