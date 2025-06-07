# src/train.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model
import mlflow
import mlflow.tensorflow

# Configuración
DATA_DIR = "data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = "models/tb_cnn_model.h5"

# 1. Cargar y preparar datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# 2. Crear modelo
model = build_model(input_shape=IMG_SIZE + (3,))

# 3. Compilar y entrenar
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Entrenamiento con MLflow
mlflow.tensorflow.autolog()  # Registro automático

with mlflow.start_run():
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

# 5. Guardar el modelo
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
print(f"Modelo guardado en: {MODEL_PATH}")
