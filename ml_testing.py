import keras
import numpy as np
import pandas as pd
from keras import models

from data_preprocessing import process_data, classifier_set


# Test the model
def test_model(model: keras.Model):
    path = './asl_alphabet_test'
    data, output_labels = process_data(path)
    data = np.asarray(data)

    all_labels = classifier_set.values()
    df = pd.get_dummies(output_labels)

    missing_labels = set(all_labels) - set(output_labels)
    for label in missing_labels:
        df[label] = 0

    result = model.predict(data)
    for i in range(len(result)):
        print("Predicted Value:", str(classifier_set[np.argmax(result[i])]))
        print("Accuracy:", str(int(max(result[i]) * 100)) + "%")
    pd.set_option('display.max_columns', None)


# Main function
def main():
    model = models.load_model('asl_recognition_model.h5')
    test_model(model)


if __name__ == '__main__':
    main()
