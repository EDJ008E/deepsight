from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from io import BytesIO
import os

app = Flask(__name__)

# Load the pre-trained model
model_path = 'deepfake_detection_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path.")
model = load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html', page="home")

@app.route('/contact')
def contact():
    return render_template('index.html', page="contact")

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the file is provided in the request
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Check for valid file format
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file format. Please upload an image file (.png, .jpg, .jpeg)'}), 400

    try:
        # Load the image
        img = load_img(BytesIO(file.read()), target_size=(150, 150))  # Resize the image to 150x150
        img_array = img_to_array(img) / 255.0  # Normalize the image to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension for prediction

        # Make a prediction
        prediction = model.predict(img_array)
        result = 'Original Image' if prediction[0] > 0.5 else 'Fake Image'

        # Return the result as JSON
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/submit-contact', methods=['POST'])
def submit_contact():
    # Get the contact form data
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')

    # Log the received message (could be saved to a database)
    print(f"Received message from {name} ({email}): {message}")

    # Render the contact page with success message
    return render_template('index.html', page="contact", success=True, name=name)

if __name__ == '__main__':
    app.run(debug=True)
