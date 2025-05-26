from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import gdown
from flask_cors import CORS
from dotenv import load_dotenv
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# initialize flask object

app = Flask(__name__)

# allow cross-orgin request sharing (CORS)

CORS(app, resources={r"/predict": {"origins": "*"}})

model_path = './plantoscope_model.tflite'
model_google_drive_url = 'https://drive.google.com/uc?id=1T0qT8ssCNPYxQtBhtHvE8-zpYSBh3GuG'

# check if the model is already loaded in the container, otherwise load it

if not os.path.exists(model_path):

    gdown.download(model_google_drive_url, model_path, quiet=True)

# load the ML model

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# load the class labels and disease information

class_indices = json.load(open('./class_indices.json'))
disease_info = json.load(open('./disease_info.json'))

# load the env (environment) files

load_dotenv('API_KEY.env')

# extract the API key from the env file

API_KEY = os.getenv('API_KEY')

# preprocess the image

def preprocess_image(image):

    img = Image.open(image).resize((224, 224))

    img_array = np.array(img) / 255.0 # normalize

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# API endpoint for home

@app.route('/')
def home():

    return 'Welcome to PlantoScope API ☘️'

# API endpoint for health check in Render (Deployment Service)

@app.route('/healthcheck')
def healthcheck():

    return jsonify({'status': 'ok'}), 200

# API endpoint for requesting the prediction

@app.route('/predict', methods=['POST'])
def predict():
    
    key = request.headers.get('x-api-key')

    if key != API_KEY:

        return jsonify({"error": "Unauthorized Request"}), 401

    if 'file' not in request.files:

        return jsonify({'error': 'No file uploaded'}), 400

    try:

        file = request.files['file']
        image = preprocess_image(file).astype(np.float32)  # TFLite expects float32

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], image)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(output_data[0])
        predicted_class_name = class_indices.get(str(predicted_class_index), 'Unknown Disease')

        if predicted_class_name != 'Unknown Disease' and '___' in predicted_class_name:

            plant_type, disease_name = predicted_class_name.split("___", 1)

        else:

            plant_type, disease_name = 'Unknown', 'Unknown'

        info = disease_info.get(predicted_class_name, {'cause': 'Unknown', 'cure': 'Unknown'})

        return jsonify({
            
            'plant_type': plant_type,
            'disease_name': disease_name,
            'cause': info['cause'],
            'cure': info['cure']

        }), 200

    except Exception as e:

        print('Prediction error:', e)

        return jsonify({

            'plant_type': 'Unknown',
            'disease_name': 'Unknown',
            'cause': 'Unknown',
            'cure': 'Unknown'

        }), 200

if __name__ == '__main__':

    app.run(host='0.0.0.0', port='8000', debug=True)
