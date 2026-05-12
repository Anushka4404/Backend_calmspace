from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from fer import FER

app = Flask(__name__)

# CORS FIX
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://frontend-calmspace.onrender.com",
            "http://localhost:3000",
            "http://localhost:5000",
            "http://localhost:5001",
            "*"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# IMPORTANT: mtcnn=False for Render free tier
detector = FER(mtcnn=False)

import os

cascade_path = os.path.join(
    os.path.dirname(__file__),
    'haarcascade_frontalface_default.xml'
)

faceDetect = cv2.CascadeClassifier(cascade_path)

labels_dict = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

@app.route('/')
def home():
    response = jsonify({"message": "ML Service Running!"})
    return add_cors_headers(response)


@app.route('/predict_emotion', methods=['OPTIONS'])
def predict_emotion_options():
    response = jsonify({})
    return add_cors_headers(response)


@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():

    try:

        if 'image' not in request.files:
            response = jsonify({'error': 'No image file'})
            response.status_code = 400
            return add_cors_headers(response)

        file = request.files['image']

        print('Received image file:', file.filename)

        npimg = np.frombuffer(file.read(), np.uint8)

        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            response = jsonify({'error': 'Invalid image'})
            response.status_code = 400
            return add_cors_headers(response)

        if frame.shape[0] == 0 or frame.shape[1] == 0:
            response = jsonify({'error': 'Empty image'})
            response.status_code = 400
            return add_cors_headers(response)

        print("Frame shape:", frame.shape)

        # MAIN FER DETECTION
        result = detector.detect_emotions(frame)

        print("Initial FER detection result:", result)

        # FALLBACK FACE DETECTION
        if not result:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceDetect.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5
            )

            print("Haar cascade faces:", faces)

            if len(faces) == 0:
                response = jsonify({'error': 'No face detected'})
                response.status_code = 400
                return add_cors_headers(response)

            x, y, w, h = faces[0]

            face_img = frame[y:y+h, x:x+w]

            result = detector.detect_emotions(face_img)

            print("FER cropped result:", result)

            if not result:
                response = jsonify({'error': 'Emotion not detected'})
                response.status_code = 400
                return add_cors_headers(response)

        emotions = result[0]["emotions"]

        # CUSTOM LOGIC
        if emotions["sad"] > 0.15:
            emotion = "sad"

        elif emotions["angry"] > 0.15:
            emotion = "angry"

        elif emotions["fear"] > 0.4:
            emotion = "sad"

        else:
            emotion = max(emotions, key=emotions.get)

        confidence = emotions.get(emotion, 0)

        labels_map = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "sad": 5,
            "surprise": 6
        }

        return add_cors_headers(jsonify({
            "mood": labels_map[emotion],
            "moodLabel": emotion.capitalize(),
            "confidence": round(confidence, 2)
        }))

    except Exception as e:

        print("ERROR:", str(e))

        response = jsonify({
            'error': str(e)
        })

        response.status_code = 500

        return add_cors_headers(response)


if __name__ == '__main__':

    port = int(os.environ.get("PORT", 10000))

    print(f"🚀 ML Server running on port {port}")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )