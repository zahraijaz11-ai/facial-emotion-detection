# ================================================================
# V24 - FACIAL EMOTION RECOGNITION
# BALANCED RAF-DB + L2 REGULARIZATION
# ================================================================

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

print("=" * 70)
print("V24 - FACIAL EMOTION RECOGNITION")
print("BALANCED RAF-DB + L2 REGULARIZATION")
print("=" * 70)

print("TensorFlow version:", tf.__version__)

# ================================================================
# GPU CHECK
# ================================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected!")
    print("GPU devices:", gpus)
else:
    print("WARNING: GPU not detected. Training will use CPU.")

# ================================================================
# CONFIGURATION
# ================================================================

IMAGE_SIZE = (75, 75)
BATCH_SIZE = 64
EPOCHS = 60

NUM_CLASSES = 7

# V24 L2
L2_REG = 0.00015

# Dropout
DROPOUT_1 = 0.15
DROPOUT_2 = 0.15
DROPOUT_3 = 0.20
DROPOUT_DENSE = 0.25

# Optimizer
LEARNING_RATE = 0.0005

# Label smoothing
LABEL_SMOOTHING = 0.05

print()
print("=" * 70)
print("CONFIGURATION")
print("=" * 70)

print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("L2 regularization:", L2_REG)
print("Learning rate:", LEARNING_RATE)
print("Label smoothing:", LABEL_SMOOTHING)

# ================================================================
# DATASET PATH
# ================================================================

DATASET_PATH = "/content/expression2.0"

TRAIN_DIR = os.path.join(
    DATASET_PATH,
    "train"
)

VAL_DIR = os.path.join(
    DATASET_PATH,
    "val"
)

TEST_DIR = os.path.join(
    DATASET_PATH,
    "test"
)

# ================================================================
# LOAD TRAIN DATASET
# ================================================================

print()
print("=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

# ================================================================
# LOAD VALIDATION DATASET
# ================================================================

print()
print("=" * 70)
print("LOADING VALIDATION DATA")
print("=" * 70)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ================================================================
# LOAD TEST DATASET
# ================================================================

print()
print("=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print()
print("Class names:")
print(train_dataset.class_names)

# ================================================================
# NORMALIZATION
# ================================================================

normalization_layer = layers.Rescaling(
    1.0 / 255
)

train_dataset = train_dataset.map(
    lambda x, y: (
        normalization_layer(x),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.map(
    lambda x, y: (
        normalization_layer(x),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.map(
    lambda x, y: (
        normalization_layer(x),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

# ================================================================
# PREFETCH
# ================================================================

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)

# ================================================================
# DATA AUGMENTATION
# ================================================================

data_augmentation = tf.keras.Sequential(
    [

        layers.RandomFlip(
            mode="horizontal"
        ),

        layers.RandomRotation(
            factor=0.08
        ),

        layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10
        ),

        layers.RandomTranslation(
            height_factor=0.10,
            width_factor=0.10
        )

    ],
    name="data_augmentation"
)

# ================================================================
# L2 REGULARIZER
# ================================================================

l2_regularizer = regularizers.l2(
    L2_REG
)

# ================================================================
# BUILD V24 CNN
# ================================================================

print()
print("=" * 70)
print("BUILDING V24 MODEL")
print("=" * 70)

model = models.Sequential(
    [

        # ========================================================
        # INPUT
        # ========================================================

        layers.Input(
            shape=(75, 75, 1)
        ),

        # ========================================================
        # DATA AUGMENTATION
        # ========================================================

        data_augmentation,

        # ========================================================
        # BLOCK 1
        # ========================================================

        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            kernel_regularizer=l2_regularizer
        ),

        layers.BatchNormalization(),

        layers.Activation(
            "relu"
        ),

        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            kernel_regularizer=l2_regularizer
        ),

        layers.BatchNormalization(),

        layers.Activation(
            "relu"
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        layers.Dropout(
            DROPOUT_1
        ),

        # ========================================================
        # BLOCK 2
        # ========================================================

        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            kernel_regularizer=l2_regularizer
        ),

        layers.BatchNormalization(),

        layers.Activation(
            "relu"
        ),

        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            kernel_regularizer=l2_regularizer
        ),

        layers.BatchNormalization(),

        layers.Activation(
            "relu"
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        layers.Dropout(
            DROPOUT_2
        ),

        # ========================================================
        # BLOCK 3
        # ========================================================

        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            kernel_regularizer=l2_regularizer
        ),

        layers.BatchNormalization(),

        layers.Activation(
            "relu"
        ),

        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            kernel_regularizer=l2_regularizer
        ),

        layers.BatchNormalization(),

        layers.Activation(
            "relu"
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        layers.Dropout(
            DROPOUT_3
        ),

        # ========================================================
        # CLASSIFIER
        # ========================================================

        layers.Flatten(),

        layers.Dense(
            256,
            kernel_regularizer=l2_regularizer
        ),

        layers.BatchNormalization(),

        layers.Activation(
            "relu"
        ),

        layers.Dropout(
            DROPOUT_DENSE
        ),

        # ========================================================
        # OUTPUT
        # ========================================================

        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )

    ],

    name="Emotion_CNN_V24"
)

# ================================================================
# MODEL SUMMARY
# ================================================================

model.summary()

# ================================================================
# COMPILE MODEL
# ================================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=LABEL_SMOOTHING
    ),

    metrics=[
        "accuracy"
    ]
)

# ================================================================
# CALLBACKS
# ================================================================

BEST_MODEL_PATH = (
    "/content/emotion_model_v24_best.keras"
)

early_stopping = EarlyStopping(

    monitor="val_accuracy",

    patience=7,

    mode="max",

    restore_best_weights=True,

    verbose=1
)

reduce_lr = ReduceLROnPlateau(

    monitor="val_accuracy",

    factor=0.5,

    patience=3,

    min_lr=1e-6,

    mode="max",

    verbose=1
)

checkpoint = ModelCheckpoint(

    BEST_MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)

# ================================================================
# TRAINING
# ================================================================

print()
print("=" * 70)
print("STARTING V24 TRAINING")
print("=" * 70)

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)

# ================================================================
# BEST VALIDATION ACCURACY
# ================================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_epoch = (
    np.argmax(
        history.history["val_accuracy"]
    ) + 1
)

print()
print("=" * 70)
print("BEST VALIDATION RESULT")
print("=" * 70)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best epoch: {best_epoch}"
)

# ================================================================
# LOAD BEST MODEL
# ================================================================

print()
print("=" * 70)
print("LOADING BEST V24 MODEL")
print("=" * 70)

best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)

# ================================================================
# TEST EVALUATION
# ================================================================

print()
print("=" * 70)
print("V24 TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = (
    best_model.evaluate(
        test_dataset,
        verbose=1
    )
)

print()
print(
    f"V24 Test Loss: "
    f"{test_loss:.4f}"
)

print(
    f"V24 Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

# ================================================================
# GENERATE PREDICTIONS
# ================================================================

print()
print("=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = best_model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        np.argmax(
            labels.numpy(),
            axis=1
        )
    )

    y_pred.extend(
        np.argmax(
            predictions,
            axis=1
        )
    )

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ================================================================
# CLASSIFICATION REPORT
# ================================================================

from sklearn.metrics import classification_report

class_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

print()
print("=" * 70)
print("V24 CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

# ================================================================
# CONFUSION MATRIX
# ================================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_true,
    y_pred
)

print()
print("=" * 70)
print("V24 CONFUSION MATRIX")
print("=" * 70)

print(cm)

# ================================================================
# FINAL SUMMARY
# ================================================================

print()
print("=" * 70)
print("V24 FINAL SUMMARY")
print("=" * 70)

print("Dataset: Balanced RAF-DB")
print("Training images: 30023")
print("Validation images: 7504")
print("Test images: 4165")

print()
print("Image size:", IMAGE_SIZE)
print("Grayscale: Yes")
print("Batch size:", BATCH_SIZE)
print("L2 regularization:", L2_REG)
print("Learning rate:", LEARNING_RATE)
print("Label smoothing:", LABEL_SMOOTHING)

print()
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print()
print("Best model saved at:")
print(BEST_MODEL_PATH)

print("=" * 70)