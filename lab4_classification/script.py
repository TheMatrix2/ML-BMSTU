import warnings
warnings.filterwarnings("ignore")

import os
from keras.src.legacy.preprocessing.image import ImageDataGenerator, DirectoryIterator
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import MaxPool2D, Conv2D, Dropout, Flatten, Dense, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32


def create_model() -> Sequential:
    # model = Sequential([
    #     Conv2D(filters=8, kernel_size=(5, 5), padding='same', activation='relu',
    #            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    #     MaxPool2D(pool_size=(2, 2)),
    #     Dropout(0.25),
    #     Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu'),
    #     MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    #     Dropout(0.25),
    #     Flatten(),
    #     Dense(256, activation='relu'),
    #     Dropout(0.5),
    #     Dense(1, activation='sigmoid')
    # ])
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        GlobalAveragePooling2D(),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def create_data_generator(dataset_path: str) -> (DirectoryIterator, DirectoryIterator, DirectoryIterator):
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    validation_dir = os.path.join(dataset_path, 'validation')

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        # validation_split=0.2
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator


def train_model(model: Sequential, train_generator: DirectoryIterator, validation_generator: DirectoryIterator, epochs: int = 30) -> Sequential:
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )

    model.fit(
        train_generator,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr],
        validation_data=validation_generator
    )

    return model


def evaluate_model(model: Sequential, test_generator: DirectoryIterator):
    loss, accuracy = model.evaluate(test_generator)
    print('Test loss:', round(loss, 4))
    print('Test accuracy:', round(accuracy, 4))

    predictions = model.predict(test_generator)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_generator.classes

    # Создаем отчет о классификации
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    print(report_df)

    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title('Матрица ошибок')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return report_df


train, validation, test = create_data_generator('dataset')
model = create_model()
model.summary()
# trained_model = train_model(model, train, validation, epochs=50)
# evaluate_model(trained_model, test)

# tf.keras.utils.plot_model(model, show_shapes=True)
