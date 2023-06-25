import pickle

import keras
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def create_model():
    model = keras.Sequential(
        [
            keras.layers.Dense(128, input_shape=(42,), activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="softmax"),
        ]
    )
    model.summary()
    return model


def train_model(model, data, labels, batch_size=128, epochs=15):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    training_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.25)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return training_history, model


def plot_learning_curve(training):
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('Learning Curve')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def main():
    data_labels = pickle.load(open('data.pickle', 'rb'))
    data = np.array(data_labels['data'])
    labels = np.array(data_labels['labels'])

    model = create_model()
    training = train_model(model, data, labels, epochs=5)
    plot_learning_curve(training)

    # Save the trained model
    model.save('emnist_merge_recognition_model')


if __name__ == '__main__':
    main()
