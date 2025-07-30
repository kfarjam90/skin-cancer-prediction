from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename  # Add this import
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load the pre-trained model
model = load_model('model/ResNet50V2_model.h5')

# Class names (must match your training classes exactly)
CLASS_NAMES = [
    'akiec',
    'bcc',
    'bkl',
    'df',
    'mel',
    'nv',
    'vasc'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess the image for model prediction"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image file")
    img = cv2.resize(img, (224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Secure the filename and save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Create upload directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)
            
            # Preprocess and predict
            processed_img = preprocess_image(filepath)
            predictions = model.predict(processed_img)[0]
            
            # Get all probabilities
            probabilities = [round(float(p) * 100, 2) for p in predictions]
            
            # Get top prediction
            predicted_class_idx = np.argmax(predictions)
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = round(float(predictions[predicted_class_idx]) * 100, 2)
            
            return render_template('result.html', 
                                prediction=predicted_class,
                                confidence=confidence,
                                class_names=CLASS_NAMES,
                                probabilities=probabilities,
                                filename=filename)
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return render_template('index.html', error=str(e))
        
    else:
        return render_template('index.html', error="Invalid file type. Allowed types: png, jpg, jpeg")

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)