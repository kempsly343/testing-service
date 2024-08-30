from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import cv2
import os

app = Flask(__name__)

# Configuration dictionary with image size and class names
CONFIGURATION = {
    "IM_SIZE": 256,
    "CLASS_NAMES": ["ACCESSORIES", "BRACELETS", "CHAIN", "CHARMS", "EARRINGS",
                    "ENGAGEMENT RINGS", "ENGAGEMENT SET", "FASHION RINGS", "NECKLACES", "WEDDING BANDS"],
}

# Global model and inference function
model = None

def init_model():
    global model
    model_path = os.path.join('models', 'update_lenet_model_save.h5')
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

def preprocess_image(image):
    # Resize and preprocess the image for the model
    image = cv2.resize(image, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))
    image = tf.convert_to_tensor(image, dtype=tf.float32)  # Ensure TensorFlow tensor
    image = image / 255.0  # Normalize the image
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the input data
        data = request.get_json()
        
        if 'image_url' in data:
            # Load image from the URL
            response = requests.get(data['image_url'])
            image = np.array(Image.open(BytesIO(response.content)).convert('RGB'))
        elif 'image_data' in data:
            # Load image from the provided image data
            image = np.array(data['image_data'], dtype=np.uint8)
        else:
            return jsonify({"error": "No valid input image provided."}), 400
        
        # Preprocess the image
        image = preprocess_image(image)
        
        # Make predictions
        predictions = model.predict(image)
        
        # Get the top 3 predicted classes and their probabilities
        predictions = predictions[0]  # Remove batch dimension
        top_3_indices = np.argsort(predictions)[-3:][::-1]  # Top 3 indices
        top_3_probabilities = predictions[top_3_indices]
        top_3_classes = [CONFIGURATION['CLASS_NAMES'][index] for index in top_3_indices]
        
        # Prepare the results as a list of dictionaries
        top_3_predictions = [
            {"class_name": top_3_classes[i], "probability": float(top_3_probabilities[i])}
            for i in range(3)
        ]
        
        # Return the JSON response
        return jsonify({"top_3_classes_predictions": top_3_predictions})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add the GET route to say "Hello"
@app.route('/', methods=['GET'])
def hello():
    return "Hello! Welcome to product type classification API with Keras."

if __name__ == '__main__':
    init_model()  # Initialize the model before starting the server
    app.run()  # Defaults to host='127.0.0.1', port=5000
