import pickle

from keras import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def create_model():
    model = Sequential(
        [
            Dense(112, input_shape=(42,), activation="relu"),
            Dense(56, activation="relu"),
            Dense(28, activation="softmax")
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model


def train_model(model, data, labels, batch_size=32, epochs=100):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
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

    df = pd.DataFrame(data={'labels': data_labels['labels']})
    categorical_dummies = pd.get_dummies(df['labels'])
    labels = np.asarray(categorical_dummies)

    model = create_model()
    training, model = train_model(model, data, labels, batch_size=128, epochs=150)
    plot_learning_curve(training)

    # Save the trained model
    model.save('asl_recognition_model.h5')


if __name__ == '__main__':
    main()
