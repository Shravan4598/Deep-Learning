# ============================================================
#              KERAS TUNER - COMPLETE TEMPLATE
# ============================================================
# Author  : Reusable Production-Level Template
# Purpose : Hyperparameter Tuning for:
#           - Binary Classification
#           - Multi-class Classification
#           - Regression
#           - ANN Projects
#           - CNN Projects
#           - Image Classification
#           - Tabular Data
#
# Tech Stack:
# TensorFlow | Keras | Keras Tuner | Scikit-learn
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

import keras_tuner as kt

# ============================================================
# 2. GPU CONFIGURATION
# ============================================================

print("TensorFlow Version:", tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Available: {len(gpus)}")
    except RuntimeError as e:
        print(e)
else:
    print("Running on CPU")


# ============================================================
# 3. RANDOM SEED FOR REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 4. CONFIGURATION SECTION
# ============================================================

# TASK TYPES:
# "binary"
# "multiclass"
# "regression"

TASK_TYPE = "binary"

# MODEL TYPES:
# "ann"
# "cnn"

MODEL_TYPE = "ann"

# ============================================================
# 5. EXAMPLE DATA LOADING SECTION
# ============================================================

# ============================================================
# OPTION 1 : CSV / TABULAR DATA
# ============================================================

USE_TABULAR_DATA = True

if USE_TABULAR_DATA:

    # Replace with your dataset path
    DATA_PATH = "your_dataset.csv"

    try:
        df = pd.read_csv(DATA_PATH)

        print(df.head())

        # ====================================================
        # TARGET COLUMN
        # ====================================================

        TARGET_COLUMN = "target"

        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN]

        # ====================================================
        # LABEL ENCODING FOR CLASSIFICATION
        # ====================================================

        if TASK_TYPE in ["binary", "multiclass"]:
            le = LabelEncoder()
            y = le.fit_transform(y)

        # ====================================================
        # TRAIN TEST SPLIT
        # ====================================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=SEED
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=SEED
        )

        # ====================================================
        # FEATURE SCALING
        # ====================================================

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        INPUT_SHAPE = X_train.shape[1]

    except Exception as e:
        print("Error Loading CSV Dataset:", e)

# ============================================================
# OPTION 2 : IMAGE DATASET
# ============================================================

USE_IMAGE_DATA = False

if USE_IMAGE_DATA:

    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32

    TRAIN_DIR = "dataset/train"
    TEST_DIR = "dataset/test"

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_dataset.class_names
    NUM_CLASSES = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    INPUT_SHAPE = (224, 224, 3)

# ============================================================
# 6. OUTPUT LAYER AUTOMATION
# ============================================================

def get_output_config(task_type):

    """
    Automatically selects output layer configuration
    based on task type.
    """

    if task_type == "binary":
        return 1, "sigmoid", "binary_crossentropy", ["accuracy"]

    elif task_type == "multiclass":
        return NUM_CLASSES, "softmax", "sparse_categorical_crossentropy", ["accuracy"]

    elif task_type == "regression":
        return 1, "linear", "mse", ["mae"]

    else:
        raise ValueError("Invalid TASK_TYPE")


# ============================================================
# 7. MODEL BUILDING FUNCTION
# ============================================================

def build_model(hp):

    """
    Hyperparameter Tuning Model Builder
    Supports:
    - ANN
    - CNN
    """

    output_units, output_activation, loss_function, metrics = get_output_config(TASK_TYPE)

    model = Sequential()

    # ========================================================
    # CNN MODEL
    # ========================================================

    if MODEL_TYPE == "cnn":

        model.add(layers.Input(shape=INPUT_SHAPE))

        # Number of CNN Blocks
        for i in range(hp.Int("conv_blocks", 1, 4)):

            filters = hp.Choice(
                f"filters_{i}",
                values=[32, 64, 128, 256]
            )

            model.add(
                layers.Conv2D(
                    filters=filters,
                    kernel_size=(3, 3),
                    padding='same',
                    activation='relu',
                    kernel_initializer=hp.Choice(
                        "kernel_initializer",
                        values=[
                            "he_normal",
                            "glorot_uniform",
                            "lecun_normal"
                        ]
                    )
                )
            )

            if hp.Boolean(f"batch_norm_{i}"):

                model.add(layers.BatchNormalization())

            model.add(layers.MaxPooling2D((2, 2)))

            dropout_rate = hp.Float(
                f"cnn_dropout_{i}",
                min_value=0.1,
                max_value=0.5,
                step=0.1
            )

            model.add(layers.Dropout(dropout_rate))

        model.add(layers.Flatten())

    # ========================================================
    # ANN MODEL
    # ========================================================

    elif MODEL_TYPE == "ann":

        model.add(layers.Input(shape=(INPUT_SHAPE,)))

    # ========================================================
    # DENSE LAYERS
    # ========================================================

    num_layers = hp.Int("num_layers", 1, 6)

    for i in range(num_layers):

        units = hp.Int(
            f"units_{i}",
            min_value=32,
            max_value=512,
            step=32
        )

        activation = hp.Choice(
            f"activation_{i}",
            values=["relu", "tanh", "selu", "elu"]
        )

        l1_value = hp.Float(
            f"l1_{i}",
            min_value=1e-6,
            max_value=1e-2,
            sampling="log"
        )

        l2_value = hp.Float(
            f"l2_{i}",
            min_value=1e-6,
            max_value=1e-2,
            sampling="log"
        )

        model.add(
            layers.Dense(
                units=units,
                activation=activation,
                kernel_initializer=hp.Choice(
                    f"initializer_{i}",
                    values=[
                        "he_normal",
                        "glorot_uniform",
                        "lecun_normal"
                    ]
                ),
                kernel_regularizer=tf.keras.regularizers.L1L2(
                    l1=l1_value,
                    l2=l2_value
                )
            )
        )

        # ====================================================
        # BATCH NORMALIZATION
        # ====================================================

        if hp.Boolean(f"use_batchnorm_{i}"):

            model.add(layers.BatchNormalization())

        # ====================================================
        # DROPOUT
        # ====================================================

        dropout_rate = hp.Float(
            f"dropout_{i}",
            min_value=0.1,
            max_value=0.5,
            step=0.1
        )

        model.add(layers.Dropout(dropout_rate))

    # ========================================================
    # OUTPUT LAYER
    # ========================================================

    model.add(
        layers.Dense(
            output_units,
            activation=output_activation
        )
    )

    # ========================================================
    # OPTIMIZER SELECTION
    # ========================================================

    learning_rate = hp.Float(
        "learning_rate",
        min_value=1e-5,
        max_value=1e-2,
        sampling="log"
    )

    optimizer_name = hp.Choice(
        "optimizer",
        values=["adam", "rmsprop", "sgd", "adamax"]
    )

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )

    elif optimizer_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate
        )

    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate
        )

    elif optimizer_name == "adamax":
        optimizer = tf.keras.optimizers.Adamax(
            learning_rate=learning_rate
        )

    # ========================================================
    # MODEL COMPILATION
    # ========================================================

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics
    )

    return model


# ============================================================
# 8. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_model.h5",
    save_best_only=True,
    monitor="val_loss"
)

# ============================================================
# 9. HYPERPARAMETER TUNER
# ============================================================

TUNER_TYPE = "hyperband"

if TUNER_TYPE == "hyperband":

    tuner = kt.Hyperband(
        build_model,
        objective="val_loss",
        max_epochs=30,
        factor=3,
        directory="keras_tuner",
        project_name="deep_learning_tuning"
    )

elif TUNER_TYPE == "bayesian":

    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_loss",
        max_trials=20,
        directory="keras_tuner",
        project_name="deep_learning_tuning"
    )

# ============================================================
# 10. TUNER SEARCH
# ============================================================

print("\nStarting Hyperparameter Search...\n")

try:

    if USE_TABULAR_DATA:

        tuner.search(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[
                early_stopping,
                reduce_lr
            ]
        )

    elif USE_IMAGE_DATA:

        tuner.search(
            train_dataset,
            validation_data=val_dataset,
            epochs=50,
            callbacks=[
                early_stopping,
                reduce_lr
            ]
        )

except Exception as e:
    print("Tuner Error:", e)

# ============================================================
# 11. BEST HYPERPARAMETERS
# ============================================================

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\nBest Hyperparameters:\n")

for param in best_hps.values:
    print(param, ":", best_hps.get(param))

# ============================================================
# 12. BUILD BEST MODEL
# ============================================================

best_model = tuner.hypermodel.build(best_hps)

best_model.summary()

# ============================================================
# 13. FINAL MODEL TRAINING
# ============================================================

EPOCHS = 100

batch_size = best_hps.get("batch_size") if "batch_size" in best_hps.values else 32

print("\nTraining Best Model...\n")

if USE_TABULAR_DATA:

    history = best_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=batch_size,
        callbacks=[
            early_stopping,
            reduce_lr,
            checkpoint
        ]
    )

elif USE_IMAGE_DATA:

    history = best_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[
            early_stopping,
            reduce_lr,
            checkpoint
        ]
    )

# ============================================================
# 14. MODEL EVALUATION
# ============================================================

print("\nEvaluating Model...\n")

if USE_TABULAR_DATA:

    results = best_model.evaluate(X_test, y_test)

elif USE_IMAGE_DATA:

    results = best_model.evaluate(test_dataset)

print("Test Results:", results)

# ============================================================
# 15. PREDICTIONS
# ============================================================

if USE_TABULAR_DATA:

    y_pred = best_model.predict(X_test)

    # ========================================================
    # BINARY CLASSIFICATION
    # ========================================================

    if TASK_TYPE == "binary":

        y_pred_classes = (y_pred > 0.5).astype(int)

        print("\nAccuracy Score:")
        print(accuracy_score(y_test, y_pred_classes))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_classes))

    # ========================================================
    # MULTICLASS CLASSIFICATION
    # ========================================================

    elif TASK_TYPE == "multiclass":

        y_pred_classes = np.argmax(y_pred, axis=1)

        print("\nAccuracy Score:")
        print(accuracy_score(y_test, y_pred_classes))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_classes))

    # ========================================================
    # REGRESSION
    # ========================================================

    elif TASK_TYPE == "regression":

        print("\nMAE:")
        print(mean_absolute_error(y_test, y_pred))

        print("\nMSE:")
        print(mean_squared_error(y_test, y_pred))

        print("\nR2 Score:")
        print(r2_score(y_test, y_pred))

# ============================================================
# 16. TRAINING CURVES
# ============================================================

plt.figure(figsize=(12, 5))

# ============================================================
# LOSS CURVE
# ============================================================

plt.subplot(1, 2, 1)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# ============================================================
# ACCURACY CURVE
# ============================================================

if TASK_TYPE in ["binary", "multiclass"]:

    plt.subplot(1, 2, 2)

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 17. SAVE MODEL
# ============================================================

# ============================================================
# H5 FORMAT
# ============================================================

best_model.save("final_model.h5")

# ============================================================
# TENSORFLOW SAVEDMODEL FORMAT
# ============================================================

best_model.export("saved_model")

print("\nModel Saved Successfully!")

# ============================================================
# 18. LOAD SAVED MODEL
# ============================================================

# H5 MODEL
loaded_model = tf.keras.models.load_model("final_model.h5")

print("\nSaved Model Loaded Successfully!")

# ============================================================
# 19. OPTIONAL: TF.DATA DATASET PIPELINE
# ============================================================

"""
Use this section for very large datasets.
Efficient for GPU training.
"""

# Example:
#
# train_ds = tf.data.Dataset.from_tensor_slices(
#     (X_train, y_train)
# )
#
# train_ds = (
#     train_ds
#     .shuffle(1000)
#     .batch(32)
#     .prefetch(tf.data.AUTOTUNE)
# )

# ============================================================
# 20. IMPORTANT NOTES
# ============================================================

"""
1. Replace dataset paths with your own.
2. Select:
   TASK_TYPE = "binary" / "multiclass" / "regression"

3. Select:
   MODEL_TYPE = "ann" / "cnn"

4. For image projects:
   USE_IMAGE_DATA = True

5. Install dependencies:

pip install tensorflow keras-tuner scikit-learn matplotlib pandas numpy

6. GPU Recommended:
   pip install tensorflow-gpu

7. Recommended For:
   - College Projects
   - Industry Projects
   - Kaggle
   - Research
   - Production Prototypes

8. Easily Extendable:
   - Transfer Learning
   - LSTM
   - GRU
   - Attention Models
   - AutoEncoders

"""

# ============================================================
# END OF TEMPLATE
# ============================================================