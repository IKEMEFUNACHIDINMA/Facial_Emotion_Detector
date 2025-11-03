from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from datetime import datetime
from io import BytesIO
from PIL import Image
import base64
import os
os.environ["TRANSFORMERS_CACHE"] = "/opt/render/project/.cache"

from model import EmotionModel

# ---------------------- CONFIG ----------------------
app = Flask(__name__)
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

# ---------------------- DATABASE ----------------------
client = MongoClient("mongodb://localhost:27017/")
db = client['emotion_detection_db']
sessions = db['sessions']

# ---------------------- MODEL ----------------------
emotion_model = EmotionModel()  # lazy load handled internally

# ---------------------- HELPERS ----------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def predict_emotion(image):
    """Predict top emotion using Hugging Face model."""
    results = emotion_model.predict(image)
    if not results:
        return None, 0.0
    top_label, conf = next(iter(results.items()))
    return top_label, conf


def save_session(name, img_b64, emotion, conf):
    """Store session in MongoDB."""
    session = {
        'name': name,
        'image': img_b64,
        'emotion_detected': emotion,
        'confidence': conf,
        'timestamp': datetime.utcnow()
    }
    return str(sessions.insert_one(session).inserted_id)


# ---------------------- ROUTES ----------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    name = request.form.get('name', 'Anonymous')

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    image = Image.open(file.stream).convert('RGB')
    emotion, conf = predict_emotion(image)
    if emotion is None:
        return jsonify({'error': 'Could not predict emotion'}), 400

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    sid = save_session(name, img_b64, emotion, conf)

    return jsonify({
        'success': True,
        'name': name,
        'emotion': emotion,
        'confidence': round(conf * 100, 2),
        'session_id': sid
    })


@app.route('/webcam', methods=['POST'])
def webcam():
    data = request.get_json()
    name, image_data = data.get('name'), data.get('image')
    if not image_data:
        return jsonify({'error': 'No image data provided'}), 400

    img_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(BytesIO(img_bytes)).convert('RGB')

    emotion, conf = predict_emotion(image)
    if emotion is None:
        return jsonify({'error': 'Could not predict emotion'}), 400

    sid = save_session(name, image_data, emotion, conf)

    return jsonify({
        'success': True,
        'name': name,
        'emotion': emotion,
        'confidence': round(conf * 100, 2),
        'session_id': sid
    })


@app.route('/history')
def history():
    records = list(sessions.find({}, {'image': 0}).sort('timestamp', -1).limit(50))
    for r in records:
        r['_id'] = str(r['_id'])
        r['timestamp'] = r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    return jsonify({'sessions': records})


# ---------------------- MAIN ----------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)




# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename
# import os
# from datetime import datetime
# from pymongo import MongoClient
# import base64
# from io import BytesIO
# from PIL import Image
# import numpy as np
# from tensorflow.keras.models import load_model
# import cv2

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# # MongoDB connection
# MONGO_URI = "mongodb://localhost:27017/"  # Change this to your MongoDB URI
# client = MongoClient(MONGO_URI)
# db = client['emotion_detection_db']
# sessions_collection = db['sessions']

# # Load the trained model
# MODEL_PATH = 'emotion_detection_model.h5'
# model = load_model(MODEL_PATH)

# # Emotion labels (adjust based on your model's training)
# EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# # Face cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def preprocess_image(image):
#     """Preprocess image for emotion detection"""
#     # Convert to grayscale
#     gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
#     # Detect face
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
#     if len(faces) == 0:
#         return None
    
#     # Get the first face detected
#     (x, y, w, h) = faces[0]
#     face_roi = gray[y:y+h, x:x+w]
    
#     # Resize to model input size (48x48 is common for emotion detection)
#     face_roi = cv2.resize(face_roi, (48, 48))
#     face_roi = face_roi.astype('float32') / 255.0
#     face_roi = np.expand_dims(face_roi, axis=0)
#     face_roi = np.expand_dims(face_roi, axis=-1)
    
#     return face_roi

# def predict_emotion(image):
#     """Predict emotion from image"""
#     preprocessed = preprocess_image(image)
    
#     if preprocessed is None:
#         return None, 0.0
    
#     # Get predictions
#     predictions = model.predict(preprocessed)
#     emotion_idx = np.argmax(predictions[0])
#     confidence = float(predictions[0][emotion_idx])
    
#     return EMOTION_LABELS[emotion_idx], confidence

# def save_to_database(name, image_data, emotion, confidence):
#     """Save detection session to MongoDB"""
#     session_data = {
#         'name': name,
#         'image': image_data,  # Base64 encoded image
#         'emotion_detected': emotion,
#         'confidence': confidence,
#         'timestamp': datetime.utcnow()
#     }
    
#     result = sessions_collection.insert_one(session_data)
#     return str(result.inserted_id)

# @app.route('/')
# def index():
#     """Render the main page"""
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     """Handle image upload and emotion detection"""
#     try:
#         # Get user name
#         name = request.form.get('name', 'Anonymous')
        
#         # Check if image is provided
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image provided'}), 400
        
#         file = request.files['image']
        
#         if file.filename == '':
#             return jsonify({'error': 'No image selected'}), 400
        
#         if file and allowed_file(file.filename):
#             # Read image
#             image = Image.open(file.stream).convert('RGB')
            
#             # Predict emotion
#             emotion, confidence = predict_emotion(image)
            
#             if emotion is None:
#                 return jsonify({'error': 'No face detected in the image'}), 400
            
#             # Convert image to base64 for storage
#             buffered = BytesIO()
#             image.save(buffered, format="JPEG")
#             img_str = base64.b64encode(buffered.getvalue()).decode()
            
#             # Save to database
#             session_id = save_to_database(name, img_str, emotion, confidence)
            
#             return jsonify({
#                 'success': True,
#                 'emotion': emotion,
#                 'confidence': round(confidence * 100, 2),
#                 'session_id': session_id,
#                 'name': name
#             })
        
#         return jsonify({'error': 'Invalid file type'}), 400
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/webcam', methods=['POST'])
# def webcam_capture():
#     """Handle webcam capture and emotion detection"""
#     try:
#         data = request.get_json()
#         name = data.get('name', 'Anonymous')
#         image_data = data.get('image')
        
#         if not image_data:
#             return jsonify({'error': 'No image data provided'}), 400
        
#         # Decode base64 image
#         image_data = image_data.split(',')[1]
#         image_bytes = base64.b64decode(image_data)
#         image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
#         # Predict emotion
#         emotion, confidence = predict_emotion(image)
        
#         if emotion is None:
#             return jsonify({'error': 'No face detected in the image'}), 400
        
#         # Save to database
#         session_id = save_to_database(name, image_data, emotion, confidence)
        
#         return jsonify({
#             'success': True,
#             'emotion': emotion,
#             'confidence': round(confidence * 100, 2),
#             'session_id': session_id,
#             'name': name
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/history')
# def history():
#     """Get detection history"""
#     try:
#         # Get last 50 sessions
#         sessions = list(sessions_collection.find().sort('timestamp', -1).limit(50))
        
#         # Convert ObjectId to string and format data
#         for session in sessions:
#             session['_id'] = str(session['_id'])
#             session['timestamp'] = session['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
#             # Don't send full image data in history list
#             session.pop('image', None)
        
#         return jsonify({'sessions': sessions})
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     # Create uploads folder if it doesn't exist
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
#     # Run the app
#     app.run(debug=True, host='0.0.0.0', port=5000)