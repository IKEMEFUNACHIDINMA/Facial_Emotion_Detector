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

