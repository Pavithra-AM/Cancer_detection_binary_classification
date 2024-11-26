import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Check if the model file exists at the given path
model_path = "C:\\Users\\HP\\Downloads\\cancer_prediction\\cancer_prediction_dataset\\cancer_detection_model.h5"
if not os.path.exists(model_path):
    print("Model file not found. Please check the path.")
    exit()

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Function to predict the class of an image
def predict_image(img_path):
    # Check if the image file exists
    if not os.path.exists(img_path):
        print(f"Image file not found: {img_path}")
        return

    img = image.load_img(img_path, target_size=(150, 150))  # Adjust image size to match the model's input
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize image

    # Predict
    prediction = model.predict(img_array)

    # Output the prediction
    if prediction[0] > 0.5:
        print("The image is classified as Malignant")
    else:
        print("The image is classified as Benign")

# Example usage - Replace with your own image path
image_path = r'C:\Users\HP\Downloads\cancer_prediction\cancer_prediction_dataset\test\image.jpg'  # Replace with a valid image path
predict_image(image_path)
