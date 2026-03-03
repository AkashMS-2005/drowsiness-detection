import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os

os.makedirs('model', exist_ok=True)

# ── Data Augmentation ──────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.1,
    brightness_range=[0.8, 1.2]
)

train_gen = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(80, 80),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='training',
    seed=42
)

val_gen = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(80, 80),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='validation',
    seed=42
)

print("Class indices:", train_gen.class_indices)

# ── CNN Model ──────────────────────────────────────────────────
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(80, 80, 1)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Callbacks ──────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        'model/drowsiness_model.h5',
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    EarlyStopping(
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

# ── Train ──────────────────────────────────────────────────────
print("\n🚀 Training started...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks
)

# ── Plot Results ───────────────────────────────────────────────
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Acc', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Acc',   color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss',   color='orange')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()

print("\n✅ Training complete!")
print(f"✅ Best val accuracy: {max(history.history['val_accuracy']):.4f}")