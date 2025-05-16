import warnings
warnings.filterwarnings("ignore")

import os
import tensorflow as tf
tf.config.list_physical_devices('GPU')
import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator, DirectoryIterator
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import MaxPool2D, Conv2D, Dropout, Activation, Flatten, Dense, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
EPOCHS = 300


def create_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=INPUT_SHAPE,
               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, (3, 3), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model


def create_data_generator(dataset_path: str) -> (DirectoryIterator, DirectoryIterator, DirectoryIterator):
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    validation_dir = os.path.join(dataset_path, 'validation')

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2
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


def train_model(model, train_generator: DirectoryIterator, validation_generator: DirectoryIterator, epochs: int = 30):
    checkpoint_save_path = 'best_model.weights.h5'
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_save_path,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
    )

    def scheduler(epoch, lr):
        if epoch < epochs // 4:
            return lr * 1.01
        elif epoch < epochs // 2:
            return lr * 1.02
        elif epoch < epochs // 4 * 3:
            return lr
        else:
            decay_factor = 0.99
            return lr * decay_factor

    lr_scheduler = LearningRateScheduler(scheduler)

    lr_reduce_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.99,
        patience=5
    )

    return model.fit(
        train_generator,
        epochs=epochs,
        callbacks=[cp_callback, lr_scheduler, lr_reduce_callback],
        validation_data=validation_generator
    )


def evaluate_model(model: Sequential, test_generator: DirectoryIterator):
    loss, accuracy = model.evaluate(test_generator)
    print('Test loss:', round(loss, 4))
    print('Test accuracy:', round(accuracy, 4))

    predictions = model.predict(test_generator)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_generator.classes

    class_names = list(test_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    print(report_df)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title('Матрица ошибок')
    plt.show()

    return report_df


def plot_training_history(history):
    """Построение графиков обучения"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Потери на обучении')
    plt.plot(history.history['val_loss'], label='Потери на валидации')
    plt.title('График потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Точность на обучении')
    plt.plot(history.history['val_accuracy'], label='Точность на валидации')
    plt.title('График точности')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_validation_curves.png')
    plt.show()


def add_prediction_to_image(image_path, model, class_names):
    """Добавление предсказания к изображению"""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]

    class_idx = 1 if prediction > 0.5 else 0
    label = class_names[class_idx]
    probability = prediction if prediction > 0.5 else 1 - prediction

    pil_img = Image.fromarray(np.uint8(img))
    draw = ImageDraw.Draw(pil_img)

    text = f"{label}: {probability * 100:.2f}%"

    try:
        font = ImageFont.load_default()
        text_width, text_height = draw.textsize(text, font=font)
        draw.rectangle([(10, 10), (20 + text_width, 20 + text_height)], fill=(255, 255, 255, 180))
        draw.text((15, 15), text, fill=(0, 0, 0), font=font)
    except AttributeError:
        draw.rectangle([(10, 10), (200, 40)], fill=(255, 255, 255, 180))
        draw.text((15, 15), text, fill=(0, 0, 0))

    return pil_img


def test_on_new_images(model, new_test_dir, class_names):
    """Визуализация предсказаний на новых изображениях"""

    if not os.path.exists(new_test_dir):
        print(f"Ошибка: директория {new_test_dir} не найдена")
        return

    image_files = [f for f in os.listdir(new_test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"В директории {new_test_dir} нет изображений")
        return

    selected_files = image_files[:4] if len(image_files) > 4 else image_files

    plt.figure(figsize=(12, 12))
    for i, file in enumerate(selected_files):
        image_path = os.path.join(new_test_dir, file)
        labeled_img = add_prediction_to_image(image_path, model, class_names)

        plt.subplot(2, 2, i + 1)
        plt.imshow(labeled_img)
        plt.title(f"Файл: {file}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('predicted_shoes.png')
    plt.show()


def main():
    dataset_path = 'dataset'

    train_generator, validation_generator, test_generator = create_data_generator(dataset_path)

    class_names = list(train_generator.class_indices.keys())
    print(f"Классы: {class_names}")

    model = create_model()
    model.summary()

    history = train_model(model, train_generator, validation_generator, epochs=EPOCHS)

    evaluate_model(model, test_generator)
    plot_training_history(history)
    test_on_new_images(model, 'images', class_names)

    model.save('shoe_classifier_model.keras')
    print("Модель сохранена как 'shoe_classifier_model.keras'")


if __name__ == "__main__":
    main()
