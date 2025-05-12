from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from PIL import Image
import io

app = Flask(__name__)



# Load the model
with open('brain_tumor_Hybrid_model.pkl', 'rb') as file:
    model_dict = pickle.load(file)
    model = model_dict['model']  # adjust this if the key is different


labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Image preprocessing function
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/')
def home():
    return render_template('index.html')  # Optional web UI

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_bytes = file.read()
    processed_img = preprocess_image(img_bytes)

    try:
        prediction = model.predict((processed_img,processed_img))

        # Threshold for multi-label classification (adjust as needed)
        threshold = 0.5
        predicted_indices = np.where(prediction[0] >= threshold)[0]

        if len(predicted_indices) == 0:
            result = ["No tumor detected (below threshold)"]
        else:
            result = [labels[i] for i in predicted_indices]

        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

    import logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)
