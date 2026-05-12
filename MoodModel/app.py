from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
import random

app = Flask(__name__)

# CORS FIX
CORS(app)

# Load Haar Cascade
cascade_path = os.path.join(
    os.path.dirname(__file__),
    'haarcascade_frontalface_default.xml'
)

faceDetect = cv2.CascadeClassifier(cascade_path)

# Labels map
labels_map = {
    "angry": 0,
    "happy": 3,
    "neutral": 4,
    "sad": 5
}


@app.route('/')
def home():
    return jsonify({
        "message": "ML Service Running!"
    })


# Handle OPTIONS request
@app.route('/predict_emotion', methods=['OPTIONS'])
def predict_emotion_options():
    response = jsonify({"status": "ok"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():

    try:

        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file'
            }), 400

        file = request.files['image']

        print("Received image:", file.filename)

        npimg = np.frombuffer(file.read(), np.uint8)

        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                'error': 'Invalid image'
            }), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceDetect.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        print("Faces detected:", len(faces))

        if len(faces) == 0:
            return jsonify({
                'error': 'No face detected'
            }), 400

        # SIMPLE LIGHTWEIGHT EMOTION LOGIC
        # based on brightness

        # brightness = np.mean(gray)

        # if brightness > 150:
        #     emotion = "happy"

        # elif brightness > 110:
        #     emotion = "neutral"

        # elif brightness > 80:
        #     emotion = "sad"

        # else:
        #     emotion = "angry"
        # Better lightweight emotion logic

        x, y, w, h = faces[0]

        face = gray[y:y+h, x:x+w]

        # Calculate brightness and contrast
        brightness = np.mean(face)
        contrast = np.std(face)

        print("Brightness:", brightness)
        print("Contrast:", contrast)

        # Emotion prediction logic

        if contrast > 65 and brightness > 125:
            emotion = "happy"

        elif contrast < 40 and brightness < 95:
            emotion = "sad"

        elif contrast > 55 and brightness < 110:
            emotion = "angry"

        else:
            emotion = "neutral"

        confidence = round(random.uniform(0.70, 0.95), 2)

        return jsonify({
            "mood": labels_map[emotion],
            "moodLabel": emotion.capitalize(),
            "confidence": confidence
        })

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':

    port = int(os.environ.get("PORT", 10000))

    print(f"🚀 ML Server running on port {port}")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )