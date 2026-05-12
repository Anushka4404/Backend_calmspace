from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import numpy as np
import cv2
import os

app = Flask(__name__)
CORS(app)

labels_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}

@app.route('/')
def home():
    return jsonify({
        "message": "ML Service Running!"
    })


@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():

    try:

        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file'
            }), 400

        file = request.files['image']

        npimg = np.frombuffer(file.read(), np.uint8)

        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                'error': 'Invalid image'
            }), 400

        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion] / 100

        return jsonify({
            "mood": labels_map.get(emotion, 4),
            "moodLabel": emotion.capitalize(),
            "confidence": round(confidence, 2)
        })

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host='0.0.0.0',
        port=port
    )