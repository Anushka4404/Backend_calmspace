from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from fer import FER
# from keras.models import load_model

app = Flask(__name__)
CORS(app)

# LOAD MODEL (no training!)
# model = load_model('model_file.h5')
detector = FER(mtcnn=True)

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file'}), 400

        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = faceDetect.detectMultiScale(gray, 1.3, 3)

        # if len(faces) == 0:
        #     return jsonify({'error': 'No face detected'}), 400

        # x, y, w, h = faces[0]
        # face = gray[y:y+h, x:x+w]
        # face = cv2.resize(face, (48,48))
        # face = face / 255.0
        # face = face.reshape(1,48,48,1)

        # prediction = model.predict(face)
        # label = int(np.argmax(prediction))

        result = detector.detect_emotions(frame)

        if not result:
            return jsonify({'error': 'No face detected'}), 400

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
        
        confidence = emotions[emotion]
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
    print("🚀 ML Server running on 5000")
    app.run(host='0.0.0.0', port=5000, debug=True)