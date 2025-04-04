from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
import gdown
from flask_cors import CORS
from dotenv import load_dotenv

# initialize flask object

app = Flask(__name__)

# allow cross-orgin request sharing (CORS)

CORS(app)

model_path = './plantoscope_model.h5'
model_google_drive_url = 'https://drive.google.com/uc?id=1UyaDOrmOKyO3hQN5c_0Ono4E3HWOBkOn'

# check if the model is already loaded in the container, otherwise load it

if not os.path.exists(model_path):

    gdown.download(model_google_drive_url, model_path, quiet=True)

# load the ML model

model = tf.keras.models.load_model(model_path)

# load the class labels and disease information

class_indices = json.load(open('./class_indices.json'))
disease_info = json.load(open('./disease_info.json'))

# load the env (environment) files

load_dotenv()

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

# API endpoint for requesting the prediction

@app.route('/predict', methods=['POST'])
def predict():

    # authenticate the request

    key = request.headers.get('x-api-key')

    if key != API_KEY:

        return jsonify({"error": "Unauthorized Request"}), 401

    # check if the image file is present in the request or not

    if 'file' not in request.files:

        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

    image = preprocess_image(file)

    # get the prediction

    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(str(predicted_class_index), 'Unknown Disease')

    # extract plant type from predicted class name

    if predicted_class_name != 'Unknown Disease' and '___' in predicted_class_name:

        plant_type, disease_name = predicted_class_name.split("___", 1)

    else:

        plant_type, disease_name = 'Unknown', 'Unknown'

    # get the disease information

    info = disease_info.get(predicted_class_name, {'cause': 'No cause available', 'cure': 'No cure available'})

    return jsonify({

        'plant_type': plant_type,
        'disease_name': disease_name,
        'cause': info['cause'],
        'cure': info['cure']
    })

if __name__ == '__main__':

    app.run(host='0.0.0.0', port='8000', debug=True)