import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.keras

# 1. Initialize MLflow Experiment
mlflow.set_experiment("Pneumonia_Detection_Project")

# 2. Generate Synthetic Chest X-Ray Data (To simulate real dataset paths smoothly)
def generate_mock_data():
    X = np.random.rand(100, 128, 128, 3).astype(np.float32)  # 100 fake 128x128 images
    y = np.random.randint(0, 2, size=(100, 1)).astype(np.float32) # Binary labels (0: Normal, 1: Pneumonia)
    return X, y

X_train, y_train = generate_mock_data()
X_val, y_val = generate_mock_data()

# 3. Start MLflow Run Tracking
with mlflow.start_run(run_name="Model_V1_Baseline"):
    
    # Define Hyperparameters to Log
    EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    
    # Log Hyperparameters into MLflow Dashboard
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("architecture", "Custom_Simple_CNN")

    # 4. Build Architecture Version 1: A basic custom CNN from scratch
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Binary output for classification
    ])

    # Compile the Network
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 5. Create custom callback to log metrics per epoch dynamically into MLflow
    class MLflowCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs:
                mlflow.log_metric("train_loss", logs['loss'], step=epoch)
                mlflow.log_metric("train_accuracy", logs['accuracy'], step=epoch)
                mlflow.log_metric("val_loss", logs['val_loss'], step=epoch)
                mlflow.log_metric("val_accuracy", logs['val_accuracy'], step=epoch)

    print("--- Starting Training for Model V1 ---")
    
    # Train the neural network
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[MLflowCallback()],
        verbose=1
    )
    
    # 6. Log and save the final trained model artifact file
    mlflow.keras.log_model(model, "model_v1_baseline_directory")
    
    # Save a local version as well for our FastAPI step later
    os.makedirs("src", exist_ok=True)
    model.save("src/pneumonia_model.h5")
    
    print("--- Model V1 Successfully Trained and Logged in MLflow! ---")
