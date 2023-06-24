import keras
from matplotlib import pyplot as plt


def create_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(47, activation="softmax"),
        ]
    )

    model.summary()
    return model


def train_model(model, data, batch_size=128, epochs=15):
    model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
    training_history = model.fit(data.x_train, data.y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    score = model.evaluate(data.x_test, data.y_test, verbose=0)
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
    model = create_model()
    # training = train_model(model, data, 128, 50)
    # plot_learning_curve(training)

    # Save the trained model
    model.save('emnist_merge_recognition_model')


if __name__ == '__main__':
    main()
