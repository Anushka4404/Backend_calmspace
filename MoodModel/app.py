from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from fer import FER
# from keras.models import load_model

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# LOAD MODEL (no training!)
# model = load_model('model_file.h5')
detector = FER(mtcnn=False)

import os

cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
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
    return "ML Service Running!"

@app.route('/predict_emotion', methods=['POST'])

def predict_emotion():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file'}), 400

        file = request.files['image']
        print('Received image file:', file.filename, file.content_type)
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        cv2.imwrite("debug.jpg", frame)

        if frame is None:
            return jsonify({'error': 'Invalid image or unsupported format'}), 400

        if frame.shape[0] == 0 or frame.shape[1] == 0:
            return jsonify({'error': 'Empty image'}), 400

        print("Frame shape:", frame.shape)
        result = detector.detect_emotions(frame)
        print("Initial FER detection result:", result)

        if not result:
            # Fallback: try Haar cascade if FER missed the face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(
                gray, 
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )  # More sensitive detection
            print("Haar cascade faces:", faces)

            if len(faces) == 0:
                return jsonify({'error': 'No face detected'}), 400

            x, y, w, h = faces[0]

            face_img = frame[y:y+h, x:x+w]

            result = detector.detect_emotions(face_img)

            print("FER cropped result:", result)

            if not result:
                return jsonify({'error': 'Emotion not detected'}), 400
            

        emotions = result[0]["emotions"]
        # 🔥 PUT YOUR LOGIC HERE
        if emotions["sad"] > 0.15:
            emotion = "sad"
        elif emotions["angry"] > 0.15:
            emotion = "angry"
        elif emotions["fear"] > 0.4:
            emotion = "sad"   # treat fear as sadness (VERY IMPORTANT)
        else:
            emotion = max(emotions, key=emotions.get)
        
        # confidence = emotions[emotion]
        confidence = emotions.get(emotion, 0)
        # map emotion → number (IMPORTANT)
        labels_map = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "sad": 5,
            "surprise": 6
        }
        return jsonify({
            "mood": labels_map[emotion],   # ✅ REQUIRED
            "moodLabel": emotion.capitalize(),
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))

    print(f"🚀 ML Server running on port {port}")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )