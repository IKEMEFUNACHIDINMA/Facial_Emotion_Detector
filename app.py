

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pymongo

app = Flask(__name__)
model = load_model('trained_models/model.h5')

# MongoDB setup (replace with your MongoDB URI)
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["emotion_detection_db"]
collection = db["emotion_records"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle image upload and prediction
    img_file = request.files['image']
    img_path = "path_to_temp_image"
    img_file.save(img_path)

    # Process image for prediction
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0

    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)
    
    # Store result in MongoDB
    collection.insert_one({"emotion_index": emotion_index})
    
    return jsonify({"emotion": int(emotion_index)})

@app.route('/webcam', methods=['POST'])
def webcam():
    data = request.get_json()
    name = data.get('name')
    image_data = data.get('image')

    # Decode base64 image
    header, encoded = image_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)

    # Save or process image, run prediction, etc.
    # Return JSON response
    return jsonify({
        "success": True,
        "name": name,
        "emotion": "Happy",
        "confidence": 0.95,
        "session_id": "abc123"
    })


if __name__ =="__main__":
    app.run(debug=True)

