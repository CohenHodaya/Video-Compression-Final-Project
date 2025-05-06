import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model



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
                if augment_data:
                    img1, img2, img3 = augment(img1, img2, img3)
                img1 = np.array(img1)
                img2 = np.array(img2)
                img3 = np.array(img3)

                x_data.append([img1, img3])
                y_data.append(img2)

    return np.array(x_data), np.array(y_data)


def build_advanced_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    skip1 = x  # Skip connection
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)

    # Skip connection
    x = layers.Concatenate()([x, skip1])

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    outputs = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def predict_and_save(model, im1_path, im3_path, save_path):
    """Load two images, predict the middle frame, and save the output (for color images)."""

    # Load images as RGB
    im1 = np.array(Image.open(im1_path).convert("RGB").resize((256, 256)))
    im3 = np.array(Image.open(im3_path).convert("RGB").resize((256, 256)))

    # Normalize the images
    im1 = im1 / 255.0
    im3 = im3 / 255.0

    # Concatenate images along the channel dimension to get shape (256,256,6)
    input_data = np.concatenate([im1, im3], axis=-1)

    # Expand dimensions to create a batch of 1: (1,256,256,6)
    input_data = np.expand_dims(input_data, axis=0)

    # Predict using the model
    predicted_img = model.predict(input_data)

    # Remove batch dimension and rescale to 0-255
    predicted_img = np.squeeze(predicted_img) * 255.0
    predicted_img = predicted_img.astype(np.uint8)

    # Save the predicted image
    output_image = Image.fromarray(predicted_img).resize((720, 1280))
    output_image.save(save_path)

    print(f"Saved predicted image at: {save_path}")

def train_model():
    parent_directory = "C:/Users/user1/Pictures/vimeo_interp_test/vimeo_interp_test/try"
    x, y = process_images(parent_directory)

    x, y = x / 255.0, y / 255.0
    x = np.concatenate([x[:, 0], x[:, 1]], axis=-1)

    model = build_advanced_cnn(x.shape[1:])
    model.fit(x, y, epochs=50, batch_size=8, validation_split=0.2, callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ])
    return model
r"""
# Example usage:
#path = rf"C:\Users\user1\Pictures\vimeo_interp_test\vimeo_interp_test\target\0006_1"
#path = rf"C:\Users\user1\Pictures\vimeo_interp_test\vimeo_interp_test\target\0006_3"
#path = rf"C:\Users\user1\Pictures\vimeo_interp_test\vimeo_interp_test\target\0001"
#path = rf"C:\Users\user1\Pictures\vimeo_interp_test\vimeo_interp_test\target\0005_4"
#path = rf"C:\Users\user1\Pictures\vimeo_interp_test\vimeo_interp_test\target\0006_4"
path = rf"C:\Users\user1\Pictures\vimeo_interp_test\vimeo_interp_test\target\1"
im1_path = rf"{path}\im1.png"
im3_path = rf"{path}\im3.png"
save_path = rf"{path}\pred17.png"
model = load_model("my_model7.keras")
"""
#model = train_model()
#predict_and_save(model, im1_path, im3_path, save_path)
#model.save("my_model7.keras")
