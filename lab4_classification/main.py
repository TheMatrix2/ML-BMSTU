# before run
# add dataset directory with content from https://www.kaggle.com/datasets/ifeanyinneji/nike-adidas-shoes-for-image-classification-dataset

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPool2D, Conv2D, Dropout, Flatten, Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Параметры изображений
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32


def create_model():
    """Создание и компиляция модели CNN"""
    model = Sequential([
        Conv2D(filters=8, kernel_size=(5, 5), padding='same', activation='relu',
               input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Changed to 1 unit for binary classification
    ])

    # Компиляция модели
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def setup_data_generators(dataset_path):
    """Настройка генераторов данных для обучения и тестирования"""
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')

    # Проверяем наличие директорий
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Ошибка: не найдены директории {train_dir} или {test_dir}")
        print("Пожалуйста, запустите сначала скрипт setup_dataset.py")
        return None, None, None

    # Генератор для обучающей выборки
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 20% для валидации
    )

    # Генератор для тестовой выборки (только нормализация)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    # Создаем генераторы
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
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


def train_model(model, train_generator, validation_generator, epochs=30):
    """Обучение модели с callbacks"""
    # Callback для ранней остановки
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Callback для уменьшения скорости обучения
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )

    # Обучение модели - Fix steps calculation to handle edge cases
    steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
    validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

    # Обучение модели
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, reduce_lr]
    )

    return history


def evaluate_model(model, test_generator):
    """Оценка модели и создание classification report"""
    # Оценка на тестовой выборке
    steps = max(1, test_generator.samples // BATCH_SIZE)
    test_loss, test_accuracy = model.evaluate(test_generator, steps=steps)
    print(f"\nТочность на тестовой выборке: {test_accuracy:.4f}")

    # Получаем предсказания
    predictions = model.predict(test_generator, steps=steps)
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


def plot_training_history(history):
    """Построение графиков обучения"""
    plt.figure(figsize=(12, 4))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Потери на обучении')
    plt.plot(history.history['val_loss'], label='Потери на валидации')
    plt.title('График потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()

    # График точности
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
    # Загрузка и предобработка изображения
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Получение предсказания
    prediction = model.predict(img_array)[0][0]

    # Определение класса и вероятности
    class_idx = 1 if prediction > 0.5 else 0
    label = class_names[class_idx]
    probability = prediction if prediction > 0.5 else 1 - prediction

    # Создание копии изображения для рисования
    pil_img = Image.fromarray(np.uint8(img))
    draw = ImageDraw.Draw(pil_img)

    # Добавление текста с предсказанием
    text = f"{label}: {probability * 100:.2f}%"

    try:
        # Если доступен шрифт
        font = ImageFont.load_default()
        text_width, text_height = draw.textsize(text, font=font)
        draw.rectangle([(10, 10), (20 + text_width, 20 + text_height)], fill=(255, 255, 255, 180))
        draw.text((15, 15), text, fill=(0, 0, 0), font=font)
    except AttributeError:
        # Альтернативный вариант, если textsize недоступен
        draw.rectangle([(10, 10), (200, 40)], fill=(255, 255, 255, 180))
        draw.text((15, 15), text, fill=(0, 0, 0))

    return pil_img


def test_on_new_images(model, dataset_path, class_names):
    """Визуализация предсказаний на новых изображениях"""
    new_test_dir = os.path.join(dataset_path, 'new_test_images')

    if not os.path.exists(new_test_dir):
        print(f"Ошибка: директория {new_test_dir} не найдена")
        return

    # Получаем список файлов
    image_files = [f for f in os.listdir(new_test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"В директории {new_test_dir} нет изображений")
        return

    # Выбираем до 4 изображений
    selected_files = image_files[:4] if len(image_files) > 4 else image_files

    # Создаем график с предсказаниями
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
    # Путь к датасету
    dataset_path = 'dataset'

    # Настраиваем генераторы данных
    train_generator, validation_generator, test_generator = setup_data_generators(dataset_path)

    if train_generator is None:
        return

    # Получаем названия классов
    class_names = list(train_generator.class_indices.keys())
    print(f"Классы: {class_names}")

    # Создаем модель
    model = create_model()
    model.summary()

    # Обучаем модель
    history = train_model(model, train_generator, validation_generator)

    # Оцениваем модель
    evaluate_model(model, test_generator)

    # Строим графики обучения
    plot_training_history(history)

    # Тестируем на новых изображениях
    test_on_new_images(model, dataset_path, class_names)

    # Сохраняем модель
    model.save('shoe_classifier_model.h5')
    print("Модель сохранена как 'shoe_classifier_model.h5'")


if __name__ == "__main__":
    main()