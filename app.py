from flask import Flask, request, jsonify,render_template
import requests
import pickle
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
import pyrebase

app = Flask(__name__)

# Load the StandardScaler and machine learning model for water quality prediction
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('model.pkl', 'rb') as model_file:
    model_water_quality = pickle.load(model_file)

# Load the trained image processing model
model_image_processing = tf.keras.models.load_model('finasih.h5')

# Firebase configuration
config = {
    "apiKey": "AIzaSyBPOYDA0ttZvGglnXKpwSCNqodThovupSM",
    "authDomain": "springjal-66c38.firebaseapp.com",
    "projectId": "springjal-66c38",
    "storageBucket": "springjal-66c38.appspot.com",
    "messagingSenderId": "989140358334",
    "appId": "1:989140358334:web:7c58efd5fd43e957297aef",
    "measurementId": "G-2XPYW8VGBG",
    "serviceAccount": "serviceAccount.json",
    "databaseURL": "https://springjal-66c38-default-rtdb.firebaseio.com/"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

@app.route('/')
def index():
    return render_template('index.html')

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

@app.route('/predict_water_quality', methods=['POST'])
def predict_water_quality():
    try:
        # Get input parameters from the request
        input_data = request.get_json(force=True)

        # Extract features and convert to float
        features = [float(input_data[param]) for param in ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']]

        # Apply StandardScaler
        scaled_features = scaler.transform([features])

        # Make predictions
        prediction = model_water_quality.predict(scaled_features)[0]

        # Convert prediction to a regular Python integer
        prediction = int(prediction)

        # Return the prediction as JSON
        return jsonify({'prediction_water_quality': prediction})

    except Exception as e:
        return jsonify({'error_water_quality': str(e)})

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
        return jsonify({'error_image_analysis': str(e)})

@app.route('/process_image', methods=['POST'])
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
        predicted_mask = model_image_processing.predict(input_array)

        # Post-process the predicted mask
        threshold = 0.4
        binary_mask = (predicted_mask > threshold).astype(np.uint8)

        # Convert the binary mask array to a PIL Image
        output_image = Image.fromarray(binary_mask[0, ..., 0] * 255)

        # Use the current timestamp to create a unique identifier
        timestamp = int(time.time())

        # Save the output image to BytesIO in JPEG format
        output_image_bytesio = BytesIO()
        output_image.save(output_image_bytesio, format='JPEG')
        output_image_bytesio.seek(0)

        # Save the BytesIO object to Firebase Storage with a unique path
        output_image_path = f'output_images/output_{timestamp}.jpg'
        storage.child(output_image_path).put(output_image_bytesio.getvalue())

        return jsonify({'output_image_url': storage.child(output_image_path).get_url(None)})
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
