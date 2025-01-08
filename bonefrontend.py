from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# Load your .h5 model
model_path = 'cnn_model.h5'  # Replace with the path to your .h5 file
model = tf.keras.models.load_model(model_path)

# Define the class names as per your dataset
class_names = ['Bone cancer', 'Osteoarthritis', 'normal', 'osteoporosis']

# Recommendations based on disease prediction
disease_recommendations = {
    'osteoporosis': 'Consume calcium-rich foods and vitamin D, engage in weight-bearing exercises, and limit caffeine and alcohol',
    'Osteoarthritis': 'Maintain a healthy weight, engage in low-impact exercise, and follow an anti-inflammatory diet',
    'Bone cancer': 'Focus on a nutrient-rich diet, stay hydrated, and discuss treatment options with healthcare providers',
    'normal': 'Maintain a balanced diet rich in calcium and vitamin D, engage in regular exercise, and avoid smoking and excessive alcohol',
}

def preprocess_image(img, target_size=(128, 128)):
    """Preprocess the image for model prediction."""
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

@app.route('/')
def home():
    return render_template('index.html')  # Render the upload form

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    processed_img = preprocess_image(img)

    # Make prediction
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_disease = class_names[predicted_class]
    recommendation = disease_recommendations.get(predicted_disease, "No specific recommendation available.")

    # Return JSON response
    return jsonify({'predicted_disease': predicted_disease, 'recommendation': recommendation})

if __name__ == '__main__':
    app.run(debug=True)
