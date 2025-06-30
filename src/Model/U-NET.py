import os
import sys
import glob
import time
import cv2
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
sys.stdout = Logger("log_26.06.25_fixed.txt")


class ImageDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator to load images on-the-fly"""

    def __init__(self, folder_paths, batch_size=16, image_size=(256, 256)):
        self.folder_paths = folder_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.indices = np.arange(len(folder_paths))

    def __len__(self):
        return len(self.folder_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_folders = [self.folder_paths[i] for i in batch_indices]

        x_batch, y_batch = [], []

        for folder_path in batch_folders:
            try:
                img1_path = os.path.join(folder_path, "im1.png")
                img2_path = os.path.join(folder_path, "im2.png")
                img3_path = os.path.join(folder_path, "im3.png")

                if all(os.path.exists(p) for p in [img1_path, img2_path, img3_path]):
                    # טעינת תמונות עם הגנה מפני NaN
                    img1 = np.array(Image.open(img1_path).convert("RGB").resize(self.image_size), dtype=np.float32) / 255.0
                    img2 = np.array(Image.open(img2_path).convert("RGB").resize(self.image_size), dtype=np.float32) / 255.0
                    img3 = np.array(Image.open(img3_path).convert("RGB").resize(self.image_size), dtype=np.float32) / 255.0

                    # וודא שאין ערכי NaN או inf
                    if (np.isnan(img1).any() or np.isnan(img2).any() or np.isnan(img3).any() or
                            np.isinf(img1).any() or np.isinf(img2).any() or np.isinf(img3).any()):
                        print(f"Warning: NaN/inf values found in {folder_path}, skipping")
                        continue

                    # Concatenate im1 and im3 as input (6 channels)
                    x_input = np.concatenate([img1, img3], axis=-1)

                    x_batch.append(x_input)
                    y_batch.append(img2)

            except Exception as e:
                print(f"Error loading {folder_path}: {e}")
                continue

        if len(x_batch) == 0:
            # אם אין דאטה תקין, החזר batch ריק
            return np.zeros((1, *self.image_size, 6), dtype=np.float32), np.zeros((1, *self.image_size, 3), dtype=np.float32)

        return np.array(x_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)

    def on_epoch_end(self):
        """Shuffle data after each epoch"""
        np.random.shuffle(self.indices)


def get_folder_paths(parent_folder):
    """Get all valid folder paths containing im1, im2, im3"""
    folder_paths = []

    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path):
            img1_path = os.path.join(folder_path, "im1.png")
            img2_path = os.path.join(folder_path, "im2.png")
            img3_path = os.path.join(folder_path, "im3.png")

            if all(os.path.exists(p) for p in [img1_path, img2_path, img3_path]):
                folder_paths.append(folder_path)

    return folder_paths


# === פונקציית האיבוד הטובה ביותר לתמונות חדות ===
def optimal_sharpness_loss(y_true, y_pred):
    """
    פונקציית איבוד מתוקנת המיועדת לתמונות חדות וברורות
    משלבת MAE, SSIM ו-Edge Loss באופן מאוזן
    """
    # הגנה מפני NaN
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Clip values to prevent numerical instability
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

    # 1. MAE Loss - טוב לחדות כללית
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    # 2. SSIM Loss - לאיכות מבנית
    ssim_val = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    ssim_loss = 1.0 - ssim_val

    # 3. Edge Loss - לחדות קצוות (גרסה מפושטת ויציבה)
    def simple_edge_loss(img_true, img_pred):
        # Sobel filters for edge detection
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)

        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])

        # Convert to grayscale for edge detection
        gray_true = tf.reduce_mean(img_true, axis=-1, keepdims=True)
        gray_pred = tf.reduce_mean(img_pred, axis=-1, keepdims=True)

        # Calculate edges
        edges_x_true = tf.nn.conv2d(gray_true, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        edges_y_true = tf.nn.conv2d(gray_true, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        edges_true = tf.sqrt(tf.square(edges_x_true) + tf.square(edges_y_true) + 1e-8)

        edges_x_pred = tf.nn.conv2d(gray_pred, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        edges_y_pred = tf.nn.conv2d(gray_pred, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        edges_pred = tf.sqrt(tf.square(edges_x_pred) + tf.square(edges_y_pred) + 1e-8)

        return tf.reduce_mean(tf.abs(edges_true - edges_pred))

    edge_loss = simple_edge_loss(y_true, y_pred)

    # משקלים מאוזנים לתמונות חדות
    total_loss = 0.6 * mae_loss + 0.25 * ssim_loss + 0.15 * edge_loss

    # הגנה נוספת מפני NaN
    total_loss = tf.where(tf.math.is_nan(total_loss), mae_loss, total_loss)

    return total_loss


def conv_block(x, filters, dropout_rate=0.1):
    """Improved conv block with dropout for better generalization"""
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x


def build_optimized_unet(input_shape):
    """
    U-Net מתוקן ומיועד לתמונות חדות
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 64, 0.1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128, 0.1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256, 0.2)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512, 0.2)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    bn = conv_block(p4, 1024, 0.3)

    # Decoder with skip connections
    u1 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn)
    u1 = layers.Concatenate()([u1, c4])
    c5 = conv_block(u1, 512, 0.2)

    u2 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u2 = layers.Concatenate()([u2, c3])
    c6 = conv_block(u2, 256, 0.2)

    u3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u3 = layers.Concatenate()([u3, c2])
    c7 = conv_block(u3, 128, 0.1)

    u4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u4 = layers.Concatenate()([u4, c1])
    c8 = conv_block(u4, 64, 0.1)

    # Final output layer with sigmoid activation
    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(c8)

    model = models.Model(inputs, outputs)

    # Custom metrics for monitoring
    def safe_psnr_metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.clip_by_value(y_true, 0.0, 1.0)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
        psnr_val = tf.image.psnr(y_true, y_pred, max_val=1.0)
        return tf.where(tf.math.is_nan(psnr_val), tf.constant(0.0), psnr_val)

    def safe_ssim_metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.clip_by_value(y_true, 0.0, 1.0)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
        ssim_val = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        return tf.where(tf.math.is_nan(ssim_val), tf.constant(0.0), ssim_val)

    # Compile with optimized settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005,  # learning rate יותר נמוך ליציבות
            clipnorm=1.0  # gradient clipping למניעת NaN
        ),
        loss=optimal_sharpness_loss,
        metrics=['mae', safe_psnr_metric, safe_ssim_metric]
    )

    return model


def predict_and_save(model, im1_path, im3_path, save_path, Size=(256, 256)):
    """Fixed prediction function"""
    try:
        im1 = np.array(Image.open(im1_path).convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
        im3 = np.array(Image.open(im3_path).convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0

        # Check for NaN values
        if np.isnan(im1).any() or np.isnan(im3).any():
            print("Warning: NaN values detected in input images")
            return

        input_data = np.concatenate([im1, im3], axis=-1)
        input_data = np.expand_dims(input_data, axis=0)

        predicted_img = model.predict(input_data)
        predicted_img = np.squeeze(predicted_img)

        # Clip and convert to uint8
        predicted_img = np.clip(predicted_img, 0, 1) * 255.0
        predicted_img = predicted_img.astype(np.uint8)

        output_image = Image.fromarray(predicted_img).resize(Size)
        output_image.save(save_path)
        print(f"Saved predicted image at: {save_path}")

    except Exception as e:
        print(f"Error in prediction: {e}")


def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.keras"))
    if not checkpoint_files:
        return None

    checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
    return checkpoint_files[-1]


def train_model(data_path, continue_training=False, batch_size=4, image_size=(256, 256)):
    """
    Train model with optimized settings to avoid NaN
    """
    # Get all folder paths
    folder_paths = get_folder_paths(data_path)
    print(f"Found {len(folder_paths)} training samples")

    if len(folder_paths) == 0:
        raise ValueError("No valid training data found!")

    # Split into train and validation
    split_idx = int(0.8 * len(folder_paths))
    train_folders = folder_paths[:split_idx]
    val_folders = folder_paths[split_idx:]

    print(f"Training samples: {len(train_folders)}")
    print(f"Validation samples: {len(val_folders)}")
    print("Using optimized sharpness loss function")

    # Create data generators
    train_gen = ImageDataGenerator(train_folders, batch_size=batch_size, image_size=image_size)
    val_gen = ImageDataGenerator(val_folders, batch_size=batch_size, image_size=image_size)

    # Setup checkpoints
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = None
    initial_epoch = 0

    if continue_training:
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"Loading checkpoint: {latest_checkpoint}")
            try:
                model = tf.keras.models.load_model(latest_checkpoint, custom_objects={
                    'optimal_sharpness_loss': optimal_sharpness_loss,
                    'safe_psnr_metric': lambda y_true, y_pred: tf.where(
                        tf.math.is_nan(tf.image.psnr(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), max_val=1.0)),
                        tf.constant(0.0),
                        tf.image.psnr(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), max_val=1.0)
                    ),
                    'safe_ssim_metric': lambda y_true, y_pred: tf.where(
                        tf.math.is_nan(tf.reduce_mean(tf.image.ssim(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), max_val=1.0))),
                        tf.constant(0.0),
                        tf.reduce_mean(tf.image.ssim(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), max_val=1.0))
                    )
                })
                initial_epoch = int(latest_checkpoint.split('_epoch_')[1].split('.')[0])
                print(f"Continuing from epoch {initial_epoch}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting fresh training")
                model = None

    if model is None:
        # Input shape: (height, width, channels) - 6 channels for concatenated images
        input_shape = (*image_size, 6)
        model = build_optimized_unet(input_shape)
        print(f"Created new optimized model with input shape: {input_shape}")

    # Improved callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:03d}.keras"),
            save_freq='epoch',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.8,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        # Callback נוסף לזיהוי NaN
        tf.keras.callbacks.TerminateOnNaN()
    ]

    # Train the model
    try:
        history = model.fit(
            train_gen,
            epochs=100,
            initial_epoch=initial_epoch,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )

        # Save final model
        model.save(os.path.join(checkpoint_dir, "final_model.keras"))
        print("Training completed successfully!")
        return model

    except Exception as e:
        print(f"Error during training: {e}")
        return model


# === הרצת אימון ===
if __name__ == "__main__":
    try:
        # טען את המודל הטוב ביותר ישירות
        best_model = load_model(rf"C:\Users\user1\PycharmProjects\FrameInterpolationModel\src\Model\best_model_FUN.keras", custom_objects={
            'optimal_sharpness_loss': optimal_sharpness_loss,
            'safe_psnr_metric': lambda y_true, y_pred: tf.where(
                tf.math.is_nan(tf.image.psnr(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), max_val=1.0)),
                tf.constant(0.0),
                tf.image.psnr(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), max_val=1.0)
            ),
            'safe_ssim_metric': lambda y_true, y_pred: tf.where(
                tf.math.is_nan(tf.reduce_mean(tf.image.ssim(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), max_val=1.0))),
                tf.constant(0.0),
                tf.reduce_mean(tf.image.ssim(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), max_val=1.0))
            )
        })

        print("Model loaded successfully!")

        # דוגמה לחיזוי
        path = rf"C:\Users\user1\Pictures\P"
        im1_path = os.path.join(path, "0000.png")
        im3_path = os.path.join(path, "0002.png")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(path, f"predicted_sharp_{timestamp}.png")
        image = cv2.imread(im3_path)
        h, w = image.shape[:2]
        size = (w, h)
        predict_and_save(best_model, im1_path, im3_path, save_path,size)

    except Exception as e:
        print(f"Error loading model or during prediction: {e}")