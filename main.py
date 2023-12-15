from flask import Flask, request, jsonify
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
import pyrebase
from io import BytesIO

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('finasih.h5')

# Firebase configuration
config = {
    "apiKey": "AIzaSyBQ8XLzlON3FK_bGGRJlw6c2CakRzKtqt0",
    "authDomain": "sih-springshed.firebaseapp.com",
    "projectId": "sih-springshed",
    "storageBucket": "sih-springshed.appspot.com",
    "messagingSenderId": "171448540338",
    "appId": "1:171448540338:web:2f3bdb3191f2261ed17dab",
    "measurementId": "G-3L2DNTEL4W",
    "serviceAccount": "serviceAccount.json",
    "databaseURL": "https://sih-springshed-default-rtdb.firebaseio.com/"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

def calculate_pixels(image_url):
    # Download the image from the URL
    response = requests.get(image_url)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to download image from the URL'})

    # Open the image
    image = Image.open(BytesIO(response.content))

    # Get the image size (width x height)
    width, height = image.size

    image = image.convert('L')

    # Calculate the total number of pixels
    total_pixels = width * height

    # Convert the image to a NumPy array for efficient pixel-wise operations
    pixels = np.array(image)

    # Calculate the number of white pixels (assuming white is represented by 255 in a grayscale image)
    white_pixels = np.count_nonzero(pixels == 255)

    return total_pixels, white_pixels

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        # Get the image URL from the request
        image_url = request.json.get('image_url', '')

        if not image_url:
            return jsonify({'error': 'Image URL is missing'})

        # Calculate pixels
        total_pixels, white_pixels = calculate_pixels(image_url)
        water_pixels = total_pixels - white_pixels
        area = water_pixels / total_pixels * 100

        result = {
            'total_pixels': total_pixels,
            'white_pixels': white_pixels,
            'water_pixels': water_pixels,
            'area': area
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the image URL from the request
        image_url = request.json['image_url']

        # Download the image from Firebase Storage
        image_response = requests.get(image_url)
        image = Image.open(BytesIO(image_response.content)).convert('L')
        image = image.resize((256, 256))
        input_array = np.array(image) / 255.0
        input_array = np.expand_dims(input_array, axis=-1)
        input_array = np.expand_dims(input_array, axis=0)

        # Make predictions
        predicted_mask = model.predict(input_array)

        # Post-process the predicted mask
        threshold = 0.4
        binary_mask = (predicted_mask > threshold).astype(np.uint8)

        # Convert the binary mask array to a PIL Image
        output_image = Image.fromarray(binary_mask[0, ..., 0] * 255)

        # Save the output image to BytesIO in JPEG format
        output_image_bytesio = BytesIO()
        output_image.save(output_image_bytesio, format='JPEG')
        output_image_bytesio.seek(0)

        # Save the BytesIO object to Firebase Storage
        output_image_path = 'output_images/output.jpg'  # Change the path as needed
        storage.child(output_image_path).put(output_image_bytesio.getvalue())

        return jsonify({'output_image_url': storage.child(output_image_path).get_url(None)})
    except Exception as e:
        return jsonify({'error': str(e)})
