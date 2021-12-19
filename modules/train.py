import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import regularizers


def prepare_datasets(validation_size):
    mfcc = pd.read_csv("../test_data/train_features/mfcc_features.csv")
    train_df = mfcc.iloc[:, :]
    X, y = train_df.iloc[:, :-1].values.reshape(
        train_df.shape[0], 13, -1), train_df.iloc[:, -1].values
    print(train_df.shape)
    print(X.shape, y.shape)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=validation_size, random_state=42)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    print(X_train.shape, y_train.shape)
    return X_train, X_validation, y_train, y_validation


def build_model(input_shape):
    model = keras.Sequential()
    # 1st conv layer
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu',
              padding='valid', strides=(1, 1), input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(
        (2, 2), strides=(1, 1), padding='same'))

    # 2nd conv layer
    model.add(keras.layers.Conv2D(64, (2, 2), strides=(
        1, 1), padding='valid', activation='relu'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), activity_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), activity_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)))
    model.add(keras.layers.Dropout(0.3))
    # output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model


def train():
    mfcc = pd.read_csv("../test_data/train_features/mfcc_features.csv")
    train_data, y = mfcc.iloc[:, :-1].values.reshape(
        mfcc.shape[0], 13, -1), mfcc.iloc[:, -1].values
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    fold = 1
    for train_index, val_index in skf.split(train_data, y):
        X_train = train_data[train_index]
        X_validation = train_data[val_index]
        y_train = y[train_index]
        y_validation = y[val_index]
        X_train = X_train[..., np.newaxis]
        X_validation = X_validation[..., np.newaxis]
        print(X_train.shape)
        print(X_validation.shape)
        print(y_train.shape)
        print(y_validation.shape)
        input_shape = (X_train.shape[1], X_train.shape[2], 1)
        model = build_model(input_shape)
        optimiser = keras.optimizers.Adam(learning_rate=0.00001)
        model.compile(optimizer=optimiser,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.summary()
        model_name = f'../weights/models/model-kfold-{fold}.h5'
        model.fit(X_train, y_train, validation_data=(
            X_validation, y_validation), batch_size=50, epochs=50)
        model.save(model_name)
        fold += 1


if __name__ == "__main__":
    train()
