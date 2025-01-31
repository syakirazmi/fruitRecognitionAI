import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model/cnn_model.h5")

# Define image dimensions
img_width, img_height = 64, 64

# Load class names from the training generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator()
train_generator = datagen.flow_from_directory(
    "dataset",  # Path to your dataset
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

# Get class names
class_names = list(train_generator.class_indices.keys())

# Function to classify an image
def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and preprocess the image
        image = Image.open(file_path)
        image = image.resize((img_width, img_height))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict the class
        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]  # Map index to class name
        confidence = np.max(prediction)

        # Display the result
        result_label.config(text=f"Predicted Class: {predicted_class_name}\nConfidence: {confidence:.2f}")

# Create the GUI
root = tk.Tk()
root.title("Image Classifier")

upload_button = tk.Button(root, text="Upload Image", command=classify_image)
upload_button.pack(pady=20)

result_label = tk.Label(root, text="Predicted Class: ", font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()