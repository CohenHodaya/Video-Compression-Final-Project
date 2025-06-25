import os
import sys
import glob
import time

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

# === הגדרת לוג ===
class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# === הפעלת הלוג ===
sys.stdout = Logger("log3.txt")

def process_images(parent_folder):
    """Extract pixel data from im1 and im3 as x, and im2 as y."""
    x_data, y_data = [], []

    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path):
            img1_path, img2_path, img3_path = [os.path.join(folder_path, f"im{i}.png") for i in [1, 2, 3]]
            if all(os.path.exists(p) for p in [img1_path, img2_path, img3_path]):
                img1 = np.array(Image.open(img1_path).convert("RGB").resize((256, 256)))
                img2 = np.array(Image.open(img2_path).convert("RGB").resize((256, 256)))
                img3 = np.array(Image.open(img3_path).convert("RGB").resize((256, 256)))
                x_data.append([img1, img3])
                y_data.append(img2)

    return np.array(x_data), np.array(y_data)

def build_advanced_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    skip1 = x
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)

    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs, outputs)
    return model

def predict_and_save(model, im1_path, im3_path, save_path,Size):
    im1 = np.array(Image.open(im1_path).convert("RGB").resize((256, 256))) / 255.0
    im3 = np.array(Image.open(im3_path).convert("RGB").resize((256, 256))) / 255.0

    input_data = np.concatenate([im1, im3], axis=-1)
    input_data = np.expand_dims(input_data, axis=0)

    predicted_img = model.predict(input_data)
    predicted_img = np.squeeze(predicted_img) * 255.0
    predicted_img = predicted_img.astype(np.uint8)

    output_image = Image.fromarray(predicted_img).resize(Size)
    output_image.save(save_path)
    print(f"Saved predicted image at: {save_path}")

def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.keras"))
    if not checkpoint_files:
        return None

    checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
    return checkpoint_files[-1]

def train_model(continue_training=False):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    parent_directory = "C:/Users/user1/Pictures/vimeo_interp_test/vimeo_interp_test/try"
    x, y = process_images(parent_directory)

    x, y = x / 255.0, y / 255.0
    x = np.concatenate([x[:, 0], x[:, 1]], axis=-1)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = None
    initial_epoch = 0

    if continue_training:
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"Loading checkpoint: {latest_checkpoint}")
            model = tf.keras.models.load_model(latest_checkpoint)
            initial_epoch = int(latest_checkpoint.split('_epoch_')[1].split('.')[0])
            print(f"Continuing from epoch {initial_epoch}")
        else:
            print("No checkpoint found, starting fresh training")

    if model is None:
        model = build_advanced_cnn(x.shape[1:])
        print("Created new model")

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:03d}.keras"),
            save_freq=10 * len(x) // 32,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    model.fit(
        x, y,
        epochs=100,
        initial_epoch=initial_epoch,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks
    )

    model.save(os.path.join(checkpoint_dir, "final_model.keras"))
    return model

# === הרצת אימון חדש מהתחלה ===
#model = train_model(continue_training=True)

# === דוגמה לחיזוי תמונה לאחר האימון (אפשר להריץ אח"כ) ===

#model = load_model("checkpoints/best_model.keras")
#path = r"C:\Users\user1\Pictures\vimeo_interp_test\vimeo_interp_test\try\0006_3"
#im1_path = os.path.join(path, "im1.png")
#im3_path = os.path.join(path, "im3.png")
#timestamp = time.strftime("%Y%m%d-%H%M%S")
#save_path = os.path.join(path, f"predicted_{timestamp}.png")
#predict_and_save(model, im1_path, im3_path, save_path)

